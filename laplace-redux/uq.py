import argparse
import yaml

import torch
import pycalib.calibration_methods
from laplace import Laplace

import utils.data_utils as du
import utils.wilds_utils as wu
import utils.utils as util
from utils.test import test
from marglik_training.train_marglik import get_backend
from baselines.swag.swag import fit_swag_and_precompute_bn_params

from copy import deepcopy

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


def main(args):
    # For Bug hunting on Cluster:
    for i in range(torch.cuda.device_count()):
        print(f"torch.cuda.mem_get_info({i}): ", torch.cuda.mem_get_info(i))
        print(f"torch.cuda.get_device_properties({i}).total_memory: ", torch.cuda.get_device_properties(i).total_memory)
        print(f"torch.cuda.memory_reserved({i}): ", torch.cuda.memory_reserved(i))
        print(f"torch.cuda.memory_allocated({i}): ", torch.cuda.memory_allocated(i))
        print(f"torch.cuda.memory_summary({i}): ", torch.cuda.memory_summary(i))


    # set device and random seed
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.prior_precision = util.get_prior_precision(args, device)
    util.set_seed(args.seed)

    # load in-distribution data
    in_data_loaders, ids, no_loss_acc = du.get_in_distribution_data_loaders(
        args, device)
    train_loader, val_loader, in_test_loader = in_data_loaders

    # fit models
    mixture_components = fit_models(args, train_loader, val_loader, device)

    # evaluate models
    metrics = evaluate_models(
        args, mixture_components, in_test_loader, ids, no_loss_acc, device)

    # save results
    util.save_results(args, metrics)



# TODO this could be put into its own File
# Copied from https://github.com/gpleiss/temperature_scaling
from torch import nn, optim
from torch.nn import functional as F
class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


# TODO this could be put into its own File
class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

# TODO this could be put into its own File
@torch.no_grad()
def IncludeTemperatureIntoLastLayer(model_T: ModelWithTemperature):
    model = deepcopy(model_T.model)
    temperature = model_T.temperature

    # get the last layer
    try:
        last_layer = list(model.children())[-1][-1] # LeNet
    except TypeError:
        try:
            last_layer = list(model.children())[-1] # WRN

            # DistilBertClassifier:
            if type(last_layer) == torch.nn.Dropout:
                last_layer = list(model.children())[-2]
        except TypeError as e:
            raise e

    # check whether last layer is Linear, otherwise it does not work
    assert type(last_layer) == torch.nn.modules.linear.Linear, f"The last layer has to be Linear, to include the temperature parameter into its weights. However it is {type(last_layer)}."

    last_layer.weight /= temperature
    last_layer.bias /= temperature

    return model



from torch import nn, optim
from torch.nn import functional as F
class _ECELoss_No_Softmax(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss_No_Softmax, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        # softmaxes = F.softmax(logits, dim=1)
        softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

# TODO this could be put into its own File
class LLLAWithHessianScaling(nn.Module):
    """ A wrapper class around an LLLA to allow tuning parameters for modifying the hessian """
    def __init__(self, model, initial_hessian_scaling_factor=1, initial_hessian_diagonal_add=0, initial_hessian_diagonal_scaling_factor=1):
        super(LLLAWithHessianScaling, self).__init__()
        self.model = model

        self._hessian_scaling_factor = initial_hessian_scaling_factor
        self._hessian_diagonal_add = initial_hessian_diagonal_add
        self._hessian_diagonal_scaling_factor = initial_hessian_diagonal_scaling_factor
        self.to(self.model._device)

    # # we enforce the constraints
    # _hessian_scaling_factor >= 1
    # _hessian_diagonal_add >= 0
    # _hessian_diagonal_scaling_factor >= 1
    # To do this, we represent the parameters in log-space and introduce an offset
    # for log to properly work, we also have to ensure that the representation is > 0 by adding a small epsilon
    @property
    def _hessian_scaling_factor(self):
        return torch.exp(self._log_hessian_scaling_factor) + 1

    @_hessian_scaling_factor.setter
    def _hessian_scaling_factor(self, new_value):
        self._log_hessian_scaling_factor = nn.Parameter(torch.log(torch.ones(1, dtype=torch.float, device=self.model._device) * new_value - 1 + 1e-7))

    @property
    def _hessian_diagonal_add(self):
        return torch.exp(self._log_hessian_diagonal_add)

    @_hessian_diagonal_add.setter
    def _hessian_diagonal_add(self, new_value):
        self._log_hessian_diagonal_add = nn.Parameter(torch.log(torch.ones(1, dtype=torch.float, device=self.model._device) * new_value + 1e-7))

    @property
    def _hessian_diagonal_scaling_factor(self):
        return torch.exp(self._log_hessian_diagonal_scaling_factor) + 1

    @_hessian_diagonal_scaling_factor.setter
    def _hessian_diagonal_scaling_factor(self, new_value):
        self._log_hessian_diagonal_scaling_factor = nn.Parameter(torch.log(torch.ones(1, dtype=torch.float, device=self.model._device) * new_value - 1 + 1e-7))


    def normal_samples(self, mean, var, n_samples, generator=None):
        """Produce samples from a batch of Normal distributions either parameterized
        by a diagonal or full covariance given by `var`.

        Parameters
        ----------
        mean : torch.Tensor
            `(batch_size, output_dim)`
        var : torch.Tensor
            (co)variance of the Normal distribution
            `(batch_size, output_dim, output_dim)` or `(batch_size, output_dim)`
        generator : torch.Generator
            random number generator
        """
        assert mean.ndim == 2, 'Invalid input shape of mean, should be 2-dimensional.'
        _, output_dim = mean.shape
        randn_samples = torch.randn((output_dim, n_samples), device=mean.device,
                                    dtype=mean.dtype, generator=generator)
        if mean.shape == var.shape:
            # diagonal covariance
            scaled_samples = var.sqrt().unsqueeze(-1) * randn_samples.unsqueeze(0)
            return (mean.unsqueeze(-1) + scaled_samples).permute((2, 0, 1))
        elif mean.shape == var.shape[:2] and var.shape[-1] == mean.shape[1]:
            # full covariance
            scale = torch.linalg.cholesky(var)
            scaled_samples = torch.matmul(scale, randn_samples.unsqueeze(0))  # expand batch dim
            return (mean.unsqueeze(-1) + scaled_samples).permute((2, 0, 1))
        else:
            raise ValueError('Invalid input shapes.')


    def scaled_predictive_samples(self, x, pred_type='glm', n_samples=100, diagonal_output=False, generator=None):
        if pred_type not in ['glm']:
            raise ValueError('Only glm supported for scaled predictive samples')

        if pred_type == 'glm':
            f_mu, f_var = self.model._glm_predictive_distribution(x)
            assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1], f_mu.shape[1]])
            if diagonal_output:
                f_var = torch.diagonal(f_var, dim1=1, dim2=2)

            new_diags_add = f_var.diagonal(dim1=1, dim2=2) * self._hessian_diagonal_scaling_factor - f_var.diagonal(dim1=1, dim2=2)
            new_diags_add = torch.diag_embed(new_diags_add, dim1=1, dim2=2)
            f_var = f_var + new_diags_add
            f_var = self._hessian_scaling_factor * f_var
            f_var = f_var + self._hessian_diagonal_add * torch.eye(f_var.shape[1], device=f_var.device).unsqueeze(0).repeat(f_var.shape[0], 1, 1)

            f_samples = self.normal_samples(f_mu, f_var, n_samples, generator)
            if self.model.likelihood == 'regression':
                return f_samples
            return torch.softmax(f_samples, dim=-1)


    def FitScalingParameters(self, val_loader, train_hessian_scaling_factor=True, train_hessian_diagonal_add=True, train_hessian_diagonal_scaling_factor=False, scaling_fitting_learning_rate=0.05, n_samples=100, max_iter=100):
        nll_criterion = nn.NLLLoss().to(self.model._device)
        ece_criterion = _ECELoss_No_Softmax().to(self.model._device)

        ###### Evaluate before fitting ######
        probs_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in val_loader:
                input, label = input.to(self.model._device), label.to(self.model._device)


                probs = self.forward(input, n_samples=n_samples)

                probs_list.append(probs)
                labels_list.append(label)
            probs = torch.cat(probs_list)
            labels = torch.cat(labels_list)

        # Calculate NLL and ECE before fitting the scaling parameters
        before_nll = nll_criterion(torch.log(probs), labels).item() # NLLLoss needs log-probabilities as input
        before_ece = ece_criterion(probs, labels).item()
        print('Before fitting scaling - NLL: %.3f, ECE: %.3f' % (before_nll, before_ece))

        # make list of parameters depending on which are specified to be trained
        params = []
        if train_hessian_scaling_factor:
            params.append(self._log_hessian_scaling_factor)
        if train_hessian_diagonal_add:
            params.append(self._log_hessian_diagonal_add)
        if train_hessian_diagonal_scaling_factor:
            params.append(self._log_hessian_diagonal_scaling_factor)

        optimizer = optim.LBFGS(params, lr=scaling_fitting_learning_rate, max_iter=max_iter)


        # Set random seed to make gradient stable
        initial_generator_seed = torch.randint(0, 10000000000000000, (1,)).item()

        def eval():
            optimizer.zero_grad()
            accum_loss = 0.0
            generator = torch.Generator(device=self.model._device)
            generator.manual_seed(initial_generator_seed)
            for input, label in val_loader:
                input, label = input.to(self.model._device), label.to(self.model._device)
                probs = self.forward(input, n_samples=n_samples, generator=generator)
                loss = nll_criterion(torch.log(probs), label)
                loss.backward()
                accum_loss += loss.item()

            print("accum_loss: ", accum_loss)
            print("hessian_scaling_factor", " (Fitted): " if train_hessian_scaling_factor else ": ", self._hessian_scaling_factor)
            print("hessian_diagonal_add", " (Fitted): " if train_hessian_diagonal_add else ": ", self._hessian_diagonal_add)
            print("hessian_diagonal_scaling_factor", " (Fitted): " if train_hessian_diagonal_scaling_factor else ": ", self._hessian_diagonal_scaling_factor)

            return accum_loss

        print("### Optimizing with LBFGS: ")
        optimizer.step(eval)

        ###### Evaluate after fitting ######
        probs_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in val_loader:
                input, label = input.to(self.model._device), label.to(self.model._device)


                probs = self.forward(input, n_samples=n_samples)

                probs_list.append(probs)
                labels_list.append(label)
            probs = torch.cat(probs_list)
            labels = torch.cat(labels_list)

        # Calculate NLL and ECE before fitting the scaling parameters
        before_nll = nll_criterion(torch.log(probs), labels).item() # NLLLoss needs log-probabilities as input
        before_ece = ece_criterion(probs, labels).item()
        print('After fitting scaling - NLL: %.3f, ECE: %.3f' % (before_nll, before_ece))

        print("### Fitted Hessian Scaling: ")
        print("hessian_scaling_factor", " (Fitted): " if train_hessian_scaling_factor else ": ", self._hessian_scaling_factor)
        print("hessian_diagonal_add", " (Fitted): " if train_hessian_diagonal_add else ": ", self._hessian_diagonal_add)
        print("hessian_diagonal_scaling_factor", " (Fitted): " if train_hessian_diagonal_scaling_factor else ": ", self._hessian_diagonal_scaling_factor)

        return self



    def forward(self, x, pred_type='glm', n_samples=100, diagonal_output=False, generator=None):
        y_prob = self.scaled_predictive_samples(x, pred_type=pred_type, n_samples=n_samples, diagonal_output=diagonal_output, generator=generator)
        y_prob = y_prob.mean(dim=0)
        return y_prob


def fit_models(args, train_loader, val_loader, device):
    """ load pre-trained weights, fit inference methods, and tune hyperparameters """

    mixture_components = list()
    for model_idx in range(args.nr_components):
        model = util.load_pretrained_model(args, model_idx, device)

        if args.use_weight_included_temperature_scaling:
            print("Including Temperature into the last layer")
            model = IncludeTemperatureIntoLastLayer(ModelWithTemperature(model).set_temperature(val_loader))


        if args.method in ['laplace', 'mola']:
            if type(args.prior_precision) is str: # file path
                prior_precision = torch.load(args.prior_precision, map_location=device)
            elif type(args.prior_precision) is float:
                prior_precision = args.prior_precision
            else:
                raise ValueError('prior precision has to be either float or string (file path)')
            Backend = get_backend(args.backend, args.approx_type)
            optional_args = dict()

            if args.subset_of_weights == 'last_layer':
                optional_args['last_layer_name'] = args.last_layer_name

            print('Fitting Laplace approximation...')
            
            model = Laplace(model, args.likelihood,
                            subset_of_weights=args.subset_of_weights,
                            hessian_structure=args.hessian_structure,
                            prior_precision=prior_precision,
                            temperature=args.temperature,
                            backend=Backend, **optional_args)
            model.fit(train_loader)

            if (args.optimize_prior_precision is not None) and (args.method == 'laplace'):
                if (type(prior_precision) is float) and (args.prior_structure != 'scalar'):
                    n = model.n_params if args.prior_structure == 'all' else model.n_layers
                    prior_precision = prior_precision * torch.ones(n, device=device)
                
                print('Optimizing prior precision for Laplace approximation...')

                verbose_prior = args.prior_structure == 'scalar'
                model.optimize_prior_precision(
                    method=args.optimize_prior_precision,
                    init_prior_prec=prior_precision,
                    val_loader=val_loader,
                    pred_type=args.pred_type,
                    link_approx=args.link_approx,
                    n_samples=args.n_samples,
                    verbose=verbose_prior
                )

            if args.use_hessian_scaling_wrapper:
                model = LLLAWithHessianScaling(model, args.hessian_scaling_factor, args.hessian_diagonal_add, args.hessian_diagonal_scaling_factor)
                max_iter = 100
                if args.benchmark == 'WILDS-amazon' and args.use_ood_val_set:
                    max_iter = 40
                model.FitScalingParameters(val_loader,
                                           train_hessian_scaling_factor=args.train_hessian_scaling_factor,
                                           train_hessian_diagonal_add=args.train_hessian_diagonal_add,
                                           train_hessian_diagonal_scaling_factor=args.train_hessian_diagonal_scaling_factor,
                                           scaling_fitting_learning_rate=args.scaling_fitting_learning_rate,
                                           n_samples=args.n_samples,
                                           max_iter=max_iter)


        elif args.method in ['swag', 'multi-swag']:
            print("Fitting SWAG...")

            model = fit_swag_and_precompute_bn_params(
                model, device, train_loader, args.swag_n_snapshots,
                args.swag_lr, args.swag_c_epochs, args.swag_c_batches, 
                args.data_parallel, args.n_samples, args.swag_bn_update_subset)

        elif (args.method == 'map' and args.likelihood == 'classification' 
              and args.use_temperature_scaling):
            print("Fitting temperature scaling model on validation data...")
            all_y_prob = [model(d[0].to(device)).detach().cpu() for d in val_loader]
            all_y_prob = torch.cat(all_y_prob, dim=0)
            all_y_true = torch.cat([d[1] for d in val_loader], dim=0)

            temperature_scaling_model = pycalib.calibration_methods.TemperatureScaling()
            temperature_scaling_model.fit(all_y_prob.numpy(), all_y_true.numpy())
            model = (model, temperature_scaling_model)

        if args.likelihood == 'regression' and args.sigma_noise is None:
            print("Optimizing noise standard deviation on validation data...")
            args.sigma_noise = wu.optimize_noise_standard_deviation(model, val_loader, device)

        mixture_components.append(model)

    return mixture_components


def evaluate_models(args, mixture_components, in_test_loader, ids, no_loss_acc, device):
    """ evaluate the models and return relevant evaluation metrics """

    metrics = []
    for i, id in enumerate(ids):
        # load test data
        test_loader = in_test_loader if i == 0 else du.get_ood_test_loader(
            args, id)

        # make model predictions and compute some metrics
        test_output, test_time = util.timing(lambda: test(
            mixture_components, test_loader, args.method, id=id,
            pred_type=args.pred_type, link_approx=args.link_approx,
            n_samples=args.n_samples, device=device, no_loss_acc=no_loss_acc,
            likelihood=args.likelihood, sigma_noise=args.sigma_noise,
            save_predictive_distributions=args.save_predictive_distributions,
            predictive_distributions_save_dir=args.predictive_distributions_save_dir))
        some_metrics, all_y_prob, all_y_var = test_output
        some_metrics['test_time'] = test_time

        if i == 0:
            all_y_prob_in = all_y_prob.clone()

        # compute more metrics, aggregate and print them:
        # log likelihood, accuracy, confidence, Brier sore, ECE, MCE, AUROC, FPR95
        more_metrics = compute_metrics(
            i, id, all_y_prob, test_loader, all_y_prob_in, all_y_var, args)
        metrics.append({**some_metrics, **more_metrics})
        print(', '.join([f'{k}: {v:.4f}' for k, v in metrics[-1].items()]))
        print("metrics: ", metrics)

    return metrics


def compute_metrics(i, id, all_y_prob, test_loader, all_y_prob_in, all_y_var, args):
    """ compute evaluation metrics """

    metrics = {}

    # compute Brier, ECE and MCE for distribution shift and WILDS benchmarks
    if (not 'OOD' in args.benchmark) and (args.benchmark != 'WILDS-poverty'):
        print(f'{args.benchmark} with distribution shift intensity {i}')
        labels = torch.cat([data[1] for data in test_loader])
        metrics['brier'] = util.get_brier_score(all_y_prob, labels)
        metrics['ece'], metrics['mce'] = util.get_calib(all_y_prob, labels)

    # compute AUROC and FPR95 for OOD benchmarks
    if args.benchmark in ['MNIST-OOD', 'FMNIST-OOD', 'CIFAR-10-OOD']:
        print(f'{args.benchmark} - dataset: {id}')
        if i > 0:
            # compute other metrics
            metrics['auroc'] = util.get_auroc(all_y_prob_in, all_y_prob)
            metrics['fpr95'], _ = util.get_fpr95(all_y_prob_in, all_y_prob)

    # compute regression calibration
    if args.benchmark == "WILDS-poverty":
        print(f'{args.benchmark} with distribution shift intensity {i}')
        labels = torch.cat([data[1] for data in test_loader])
        metrics['calib_regression'] = util.get_calib_regression(
            all_y_prob.numpy(), all_y_var.sqrt().numpy(), labels.numpy())

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str,
                        choices=['R-MNIST', 'R-FMNIST', 'CIFAR-10-C', 'ImageNet-C',
                                 'MNIST-OOD', 'FMNIST-OOD', 'CIFAR-10-OOD',
                                 'WILDS-camelyon17', 'WILDS-iwildcam',
                                 'WILDS-civilcomments', 'WILDS-amazon',
                                 'WILDS-fmow', 'WILDS-poverty', 'SkinLesions', 'HAM10000-C'],
                        default='CIFAR-10-C', help='name of benchmark')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='root of dataset')
    parser.add_argument('--download', action='store_true',
                        help='if True, downloads the datasets needed for given benchmark')
    parser.add_argument('--data_fraction', type=float, default=1.0,
                    help='fraction of data to use (only supported for WILDS)')
    parser.add_argument('--models_root', type=str, default='./models',
                        help='root of pre-trained models')
    parser.add_argument('--model_seed', type=int, default=None,
                        help='random seed with which model(s) were trained')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--hessians_root', type=str, default='./hessians',
                        help='root of pre-computed Hessians')
    parser.add_argument('--method', type=str,
                        choices=['map', 'ensemble',
                                 'laplace', 'mola',
                                 'swag', 'multi-swag',
                                 'bbb', 'csghmc'],
                        default='laplace',
                        help='name of method to use')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')

    parser.add_argument('--pred_type', type=str,
                        choices=['nn', 'glm'],
                        default='glm',
                        help='type of approximation of predictive distribution')
    parser.add_argument('--link_approx', type=str,
                        choices=['mc', 'probit', 'bridge'],
                        default='probit',
                        help='type of approximation of link function')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='nr. of MC samples for approximating the predictive distribution')

    parser.add_argument('--likelihood', type=str, choices=['classification', 'regression'],
                        default='classification', help='likelihood for Laplace')
    parser.add_argument('--subset_of_weights', type=str, choices=['last_layer', 'all'],
                        default='last_layer', help='subset of weights for Laplace')
    parser.add_argument('--backend', type=str, choices=['backpack', 'kazuki'], default='backpack')
    parser.add_argument('--approx_type', type=str, choices=['ggn', 'ef'], default='ggn')
    parser.add_argument('--hessian_structure', type=str, choices=['diag', 'kron', 'full'],
                        default='kron', help='structure of the Hessian approximation')
    parser.add_argument('--last_layer_name', type=str, default=None,
                        help='name of the last layer of the model')
    parser.add_argument('--prior_precision', default=1.,
                        help='prior precision to use for computing the covariance matrix')
    parser.add_argument('--optimize_prior_precision', default=None,
                        choices=['marglik', 'nll'],
                        help='optimize prior precision according to specified method')
    parser.add_argument('--prior_structure', type=str, default='scalar',
                        choices=['scalar', 'layerwise', 'all'])
    parser.add_argument('--sigma_noise', type=float, default=None,
                        help='noise standard deviation for regression (if -1, optimize it)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature of the likelihood.')

    parser.add_argument('--swag_n_snapshots', type=int, default=40,
                        help='number of snapshots for [Multi]SWAG')
    parser.add_argument('--swag_c_batches', type=int, default=None,
                        help='number of batches between snapshots for [Multi]SWAG')
    parser.add_argument('--swag_c_epochs', type=int, default=1,
                        help='number of epochs between snapshots for [Multi]SWAG')
    parser.add_argument('--swag_lr', type=float, default=1e-2,
                        help='learning rate for [Multi]SWAG')
    parser.add_argument('--swag_bn_update_subset', type=float, default=1.0,
                        help='fraction of train data for updating the BatchNorm statistics for [Multi]SWAG')

    parser.add_argument('--nr_components', type=int, default=1,
                        help='number of mixture components to use')
    parser.add_argument('--mixture_weights', type=str,
                        choices=['uniform', 'optimize'],
                        default='uniform',
                        help='how the mixture weights for MoLA are chosen')

    parser.add_argument('--model', type=str, default='WRN16-4',
                        choices=['LeNet', 'WRN16-4', 'WRN16-4-fixup', 'WRN50-2',
                                 'LeNet-BBB-reparam', 'LeNet-BBB-flipout', 'LeNet-CSGHMC',
                                 'WRN16-4-BBB-reparam', 'WRN16-4-BBB-flipout', 'WRN16-4-CSGHMC', 'resnet50'],
                         help='the neural network model architecture')
    parser.add_argument('--no_dropout', action='store_true', help='only for WRN-fixup.')
    parser.add_argument('--data_parallel', action='store_true',
                        help='if True, use torch.nn.DataParallel(model)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch size for testing')
    parser.add_argument('--val_set_size', type=int, default=2000,
                        help='size of validation set (taken from test set)')
    parser.add_argument('--use_temperature_scaling', default=False,
                        help='if True, calibrate model using temperature scaling')

    parser.add_argument('--job_id', type=int, default=0,
                        help='job ID, leave at 0 when running locally')
    parser.add_argument('--config', default=None, nargs='+',
                        help='YAML config file path')
    parser.add_argument('--run_name', type=str, help='overwrite save file name')
    parser.add_argument('--noda', action='store_true')
    parser.add_argument('--use_weight_included_temperature_scaling', action='store_true',
                        help='if True, calibrate model using temperature scaling and inlcude this temperature into the parameters of the last layer to allow use of TS with Laplace')
    parser.add_argument('--save_predictive_distributions', action='store_true',
                        help='if True, save samples and model predictive distributions for post-hoc analysis of posteriors')
    parser.add_argument('--predictive_distributions_save_dir', type=str,
                        default='results/predictive_distributions/default/', help='the file in which the predictions are saved')
    parser.add_argument('--use_hessian_scaling_wrapper', action='store_true',
                        help='if True, use a wrapper class to implement hessian scaling ')
    parser.add_argument('--hessian_scaling_factor', type=float, default=1, help='a factor, by which the hessian is scaled')
    parser.add_argument('--hessian_diagonal_add', type=float, default=0, help='a value that is added to the diagonal of the hessian')
    parser.add_argument('--hessian_diagonal_scaling_factor', type=float, default=1, help='a value by which the diagonal of the hessian is scaled')
    parser.add_argument('--train_hessian_scaling_factor', action='store_true',
                        help='if True, Train this Parameter')
    parser.add_argument('--train_hessian_diagonal_add', action='store_true',
                        help='if True, Train this Parameter')
    parser.add_argument('--train_hessian_diagonal_scaling_factor', action='store_true',
                        help='if True, Train this Parameter')
    parser.add_argument('--scaling_fitting_learning_rate', type=float, default=0.05, help='The learning rate that is used for fitting the scaling parameters with LBFGS')
    parser.add_argument('--use_ood_val_set', action='store_true',
                        help='if True, use the OOD val set instead of the ID val set, e.g. for fitting the Temperature, Laplace prior precision or scaling parameters, if the respective parameters are fitted (only possible for WILDS datasets with OOD-val set)')
    parser.add_argument('--specific_ablation_model', type=str, default='', choices=['skinlesions_wrn50', 'camelyon17_resnet50', 'camelyon17_wrn50'], help='specify a specific model for ablation') # TODO add more choices # TODO only make change of image_size possible if needed


    args = parser.parse_args()
    args_dict = vars(args)

    # load config file (YAML)
    if args.config is not None:
        for path in args.config:
            with open(path) as f:
                config = yaml.full_load(f)
            args_dict.update(config)

    try:
        args.prior_precision = float(args.prior_precision)
    except ValueError:
        pass

    if args.data_parallel and (args.method in ['laplace, mola']):
        raise NotImplementedError(
            'laplace and mola do not support DataParallel yet.')

    if (args.optimize_prior_precision is not None) and (args.method == 'mola'):
        raise NotImplementedError(
            'optimizing the prior precision for MoLA is not supported yet.')

    if args.mixture_weights != 'uniform':
        raise NotImplementedError(
            'Only uniform mixture weights are supported for now.')

    if ((args.method in ['ensemble', 'mola', 'multi-swag']) 
        and (args.nr_components <= 1)):
        parser.error(
            'Choose nr_components > 1 for ensemble, MoLA, or MultiSWAG.')

    if args.model != 'WRN16-4-fixup' and args.no_dropout:
        parser.error(
            'No dropout option only available for Fixup.')

    if args.benchmark in ['R-MNIST', 'MNIST-OOD', 'R-FMNIST', 'FMNIST-OOD']:
        if 'LeNet' not in args.model:
            parser.error("Only LeNet works for R-MNIST.")
    elif args.benchmark in ['CIFAR-10-C', 'CIFAR-10-OOD']:
        if 'WRN16-4' not in args.model:
            parser.error("Only WRN16-4 works for CIFAR-10-C.")
    elif args.benchmark == 'ImageNet-C':
        if not (args.model == 'WRN50-2' or args.model == "resnet50"):
            parser.error("Only WRN50-2 or resnet50 works for ImageNet-C.")

    if args.benchmark == "WILDS-poverty":
        args.likelihood = "regression"
    else:
        args.likelihood = "classification"

    if args.use_weight_included_temperature_scaling and args.method not in ['map', 'laplace']:
        parser.error('Weight included temperature scaling only works with `map` or `laplace`')

    if args.save_predictive_distributions:
        if not args.method == 'laplace' or not args.link_approx == 'mc':
            parser.error('Saving the predictive distributions only works for `laplace` with `mc`')

    if args.use_ood_val_set:
        if not ('WILDS-camelyon17' in args.benchmark or 'WILDS-amazon' in args.benchmark):
            parser.error('Using an OOD val set instead of an ID val set is currently only tested for WILDS-camelyon17 and WILDS-amazon. In general, the dataset needs an OOD val set for this to work')

    for key, val in args_dict.items():
        print(f'{key}: {val}')
    print()

    main(args)
