from tqdm import tqdm
import torch
from torch import nn

from utils.utils import mixture_model_pred


def batch_cov(points):
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)


def normal_samples(mean, var, n_samples, generator=None):
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


def predictive_samples(model, x, pred_type='glm', n_samples=100, diagonal_output=False, generator=None):
    if pred_type not in ['glm']:
        raise ValueError('Only glm supported for scaled predictive samples')

    if pred_type == 'glm':
        f_mu, f_var = model._glm_predictive_distribution(x)
        assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1], f_mu.shape[1]])
        if diagonal_output:
            f_var = torch.diagonal(f_var, dim1=1, dim2=2)
        f_samples = normal_samples(f_mu, f_var, n_samples, generator)
        if model.likelihood == 'regression':
            return f_samples
        return torch.softmax(f_samples, dim=-1)


def get_predictive_distribution_parameters(model, x, pred_type='glm', diagonal_output=False):
    # Get the parameters of the distribution (f_mu, f_var) over the output logits
    # this is not very efficient, as it has to be evaluated a second time
    # -> It should be refactored at some point
    if pred_type not in ['glm']:
        raise ValueError('Only glm supported for get predictive distribution parameters')

    if pred_type == 'glm':
        if str(type(model)) == "<class '__main__.LLLAWithHessianScaling'>":
            f_mu, f_var = model.model._glm_predictive_distribution(x)
            assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1], f_mu.shape[1]])
            if diagonal_output:
                f_var = torch.diagonal(f_var, dim1=1, dim2=2)

            new_diags_add = f_var.diagonal(dim1=1, dim2=2) * model._hessian_diagonal_scaling_factor - f_var.diagonal(dim1=1, dim2=2)
            new_diags_add = torch.diag_embed(new_diags_add, dim1=1, dim2=2)
            f_var = f_var + new_diags_add
            f_var = model._hessian_scaling_factor * f_var
            f_var = f_var + model._hessian_diagonal_add * torch.eye(f_var.shape[1], device=f_var.device).unsqueeze(0).repeat(f_var.shape[0], 1, 1)


        else:
            f_mu, f_var = model._glm_predictive_distribution(x)
            assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1], f_mu.shape[1]])
            if diagonal_output:
                f_var = torch.diagonal(f_var, dim1=1, dim2=2)

        return f_mu, f_var


@torch.no_grad()
def test(components, test_loader, prediction_mode, id, pred_type='glm', n_samples=100,
         link_approx='probit', no_loss_acc=False, device='cpu',
         likelihood='classification', sigma_noise=None, verbose=True, save_predictive_distributions=False, predictive_distributions_save_dir=None):

    temperature_scaling_model = None
    if prediction_mode in ['map', 'laplace', 'bbb', 'csghmc']:
        model = components[0]
        if prediction_mode in ['map', 'bbb']:
            if prediction_mode == 'map' and isinstance(model, tuple):
                model, temperature_scaling_model = model[0], model[1]
            model.eval()
        elif prediction_mode == 'csghmc':
            for m in model:
                m.eval()
    elif prediction_mode == 'swag':
        model, swag_samples, swag_bn_params = components[0]

    if likelihood == 'regression' and sigma_noise is None:
        raise ValueError('Must provide sigma_noise for regression!')

    if likelihood == 'classification':
        loss_fn = nn.NLLLoss()
    elif likelihood == 'regression':
        loss_fn = nn.GaussianNLLLoss(full=True)
    else:
        raise ValueError(f'Invalid likelihood type {likelihood}')

    all_y_true = list()
    all_y_prob = list()
    all_y_var = list()
    all_covariances = list()

    if save_predictive_distributions:
        y_true_list = []
        f_mu_list = []
        f_var_list = []

    for data in tqdm(test_loader) if verbose else test_loader:
        x, y = data[0].to(device), data[1].to(device)
        all_y_true.append(y.cpu())

        if save_predictive_distributions:
            y_true_list.append(y.cpu())

        if prediction_mode in ['ensemble', 'mola', 'multi-swag']:
            # set uniform mixture weights
            K = len(components)
            pi = torch.ones(K, device=device) / K
            y_prob = mixture_model_pred(
                components, x, pi,
                prediction_mode=prediction_mode,
                pred_type=pred_type,
                link_approx=link_approx,
                n_samples=n_samples,
                likelihood=likelihood)

        elif prediction_mode == 'laplace':
            if link_approx == 'mc':
                if str(type(model)) == "<class '__main__.LLLAWithHessianScaling'>":
                    y_prob = model.scaled_predictive_samples(x, pred_type=pred_type, n_samples=n_samples)
                else:
                    y_prob = model.predictive_samples(x, pred_type=pred_type, n_samples=n_samples)

                y_prob = y_prob.detach()

                if save_predictive_distributions:
                    f_mu, f_var = get_predictive_distribution_parameters(model, x, pred_type=pred_type)
                    f_mu_list.append(f_mu.cpu())
                    f_var_list.append(f_var.cpu())

                covariances = batch_cov(y_prob.permute(1,0,2))
                all_covariances.append(covariances.cpu())
                y_prob = y_prob.mean(dim=0)

            else:
                y_prob = model(
                    x, pred_type=pred_type, link_approx=link_approx, n_samples=n_samples)

        elif prediction_mode == 'map':
            y_prob = model(x).detach()

        elif prediction_mode == 'bbb':
            y_prob = torch.stack([model(x)[0].softmax(-1) for _ in range(10)]).mean(0)

        elif prediction_mode == 'csghmc':
            y_prob = torch.stack([m(x).softmax(-1) for m in model]).mean(0)

        elif prediction_mode == 'swag':
            from baselines.swag.swag import predict_swag
            y_prob = predict_swag(model, x, swag_samples, swag_bn_params)

        else:
            raise ValueError(
                'Choose one out of: map, ensemble, laplace, mola, bbb, csghmc, swag, multi-swag.')

        if likelihood == 'regression':
            y_mean = y_prob if prediction_mode == 'map' else y_prob[0]
            y_var = torch.zeros_like(y_mean) if prediction_mode == 'map' else y_prob[1].squeeze(2)
            all_y_prob.append(y_mean.cpu())
            all_y_var.append(y_var.cpu())
        else:
            all_y_prob.append(y_prob.cpu())


    if save_predictive_distributions:
        y_true_list = torch.cat(y_true_list, dim=0)
        f_mu_list = torch.cat(f_mu_list, dim=0)
        f_var_list = torch.cat(f_var_list, dim=0)

        import os
        if not os.path.exists(predictive_distributions_save_dir):
            os.makedirs(predictive_distributions_save_dir)

        torch.save(y_true_list, predictive_distributions_save_dir + "y_true_" + str(id) + ".pt")
        torch.save(f_mu_list, predictive_distributions_save_dir + "f_mu_" + str(id) + ".pt")
        torch.save(f_var_list, predictive_distributions_save_dir + "f_var_" + str(id) + ".pt")

    # aggregate predictive distributions, true labels and metadata
    all_y_prob = torch.cat(all_y_prob, dim=0)
    all_y_true = torch.cat(all_y_true, dim=0)

    if temperature_scaling_model is not None:
        print('Calibrating predictions using temperature scaling...')
        all_y_prob = torch.from_numpy(temperature_scaling_model.predict_proba(all_y_prob.numpy()))

    elif prediction_mode == 'map' and likelihood == 'classification':
        all_y_prob = all_y_prob.softmax(dim=1)

    # compute some metrics: mean confidence, accuracy and negative log-likelihood
    metrics = {}
    if likelihood == 'classification':
        assert all_y_prob.sum(-1).mean() == 1, '`all_y_prob` are logits but probs. are required'
        c, preds = torch.max(all_y_prob, 1)
        metrics['conf'] = c.mean().item()

        if all_covariances:
            all_covariances = torch.cat(all_covariances, dim=0)
            variances = torch.tensor([c[preds[i], preds[i]] for i, c in enumerate(covariances)])
            metrics["mean_variance"] = variances.mean().item()


    if not no_loss_acc:
        if likelihood == 'regression':
            all_y_var = torch.cat(all_y_var, dim=0) + sigma_noise**2
            metrics['nll'] = loss_fn(all_y_prob, all_y_true, all_y_var).item()

        else:
            all_y_var = None
            metrics['nll'] = loss_fn(all_y_prob.log(), all_y_true).item()
            metrics['acc'] = (all_y_true == preds).float().mean().item()

    return metrics, all_y_prob, all_y_var


@torch.no_grad()
def predict(dataloader, model):
    py = []

    for x, y in dataloader:
        x = x.cuda()
        py.append(torch.softmax(model(x), -1))

    return torch.cat(py, dim=0)


@torch.no_grad()
def predict_ensemble(dataloader, models):
    py = []

    for x, y in dataloader:
        x = x.cuda()

        _py = 0
        for model in models:
            _py += 1/len(models) * torch.softmax(model(x), -1)
        py.append(_py)

    return torch.cat(py, dim=0)


@torch.no_grad()
def predict_vb(dataloader, model, n_samples=1):
    py = []

    for x, y in dataloader:
        x = x.cuda()

        _py = 0
        for _ in range(n_samples):
            f_s, _ = model(x)  # The second return is KL
            _py += torch.softmax(f_s, 1)
        _py /= n_samples

        py.append(_py)

    return torch.cat(py, dim=0)
