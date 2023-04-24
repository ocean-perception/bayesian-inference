import numpy as np
import torch


def calc_auxiliary_target_distribution(mat_soft_assignment):
    # auxiliary target distribution. n_samples * n_classes
    num_samples = mat_soft_assignment.size()[0]
    num_classes = mat_soft_assignment.size()[1]

    # soft cluster frequency
    freq = torch.sum(mat_soft_assignment, dim=0)  # 1 * n_classes
    q_pow = mat_soft_assignment**2  # n_samples * n_classes

    # TODO vectorization
    numerator = torch.zeros((num_samples, num_classes), dtype=torch.double)
    for idx_j in range(num_classes):
        numerator[:, idx_j] = q_pow[:, idx_j] / freq[idx_j]

    denominator = torch.zeros((num_samples, num_classes), dtype=torch.double)
    for idx_i in range(num_samples):
        denominator[idx_i, :] = torch.sum(numerator[idx_i, :])

    aux_tar_dist = numerator / denominator

    return aux_tar_dist


def calc_soft_assignment(samples, centroids, alpha=1.0):
    """

    :param samples: num_samples * num_features
    :param centroids: num_classes * num_features
    :param alpha:
    :return:
    """
    assert (
        samples.size()[1] == centroids.size()[1]
    ), "num_features should be the same for samples and centroids"

    samples = samples.double()
    centroids = centroids.double()

    num_samples = samples.size()[0]
    num_classes = centroids.size()[0]
    # num_features = samples.size()[1]

    power_value = -(alpha + 1.0) / 2.0
    soft_assignment = torch.zeros((num_samples, num_classes), dtype=torch.double)
    for idx_i in range(num_samples):
        tmp_numerator = torch.zeros((1, num_classes), dtype=torch.double)
        for idx_j in range(num_classes):
            # original
            # tmp_numerator[0, idx_j] = \
            #     (1.0 + torch.sum((samples[idx_i] - centroids[idx_j]) ** 2.0) / alpha) \
            #     ** power_value

            # with normalization
            normalize_term = 0.01
            normalize_term = 1.0
            tmp_numerator[0, idx_j] = (
                1.0
                + torch.sum(
                    ((samples[idx_i] - centroids[idx_j]) / normalize_term) ** 2.0
                )
                / alpha
            ) ** power_value

            # GMM style
            # tmp_numerator[0,idx_j]=

        tmp_denominator = torch.sum(tmp_numerator)
        soft_assignment[idx_i, :] = tmp_numerator / tmp_denominator

    return soft_assignment


def calc_kld(soft_assignment, aux_tar_dist):
    assert (
        soft_assignment.size() == aux_tar_dist.size()
    ), "shape of soft_assignment and aux_tar_dist must be the same"

    # p: auxiliary target distribution
    # q: softmax assignment
    kld = torch.sum(aux_tar_dist * torch.log(aux_tar_dist / soft_assignment))

    return kld


def calc_dec_loss(samples, centroids, alpha=1.0):
    assert (
        samples.size()[1] == centroids.size()[1]
    ), "num_features should be the same for samples and centroids"

    # num_samples = samples.size()[0]
    # num_classes = centroids.size()[0]
    # num_features = samples.size()[1]

    q = calc_soft_assignment(samples, centroids, alpha=alpha)
    q = q.cuda()
    p = calc_auxiliary_target_distribution(q)
    p = p.cuda()
    loss = calc_kld(q, p)
    if loss < 0:
        print("DEBUG loss =", loss)
    loss = loss.cuda()

    return loss


def calc_d_loss_d_z(samples, centroids, alpha=1.0):
    num_samples = samples.size()[0]
    num_classes = centroids.size()[0]
    num_features = samples.size()[1]

    q = calc_soft_assignment(samples, centroids, alpha=alpha)
    q = q.cuda()
    p = calc_auxiliary_target_distribution(q)
    p = p.cuda()

    # for calculating derivative
    p_min_q = p - q  # num_samples * num_classes
    p_min_q = p_min_q.cuda()

    # TODO implement
    d_loss_d_z = torch.zeros((num_samples, num_features), dtype=torch.double).cuda()
    for idx_sample in range(num_samples):
        tmp_sigma = torch.zeros((1, num_features), dtype=torch.double).cuda()
        for idx_k in range(num_classes):
            tmp_term_1 = (
                1.0
                + torch.sum((samples[idx_sample, :] - centroids[idx_k, :]) ** 2) / alpha
            ) ** (-1.0)
            tmp_term_2 = p[idx_sample, idx_k] - q[idx_sample, idx_k]
            tmp_term_3 = samples[idx_sample, :] - centroids[idx_k, :]
            tmp_sigma += (tmp_term_1 * tmp_term_2 * tmp_term_3).view(1, -1)

        d_loss_d_z[idx_sample] = ((alpha + 1.0) / alpha * tmp_sigma).cuda()

    return d_loss_d_z


def calc_d_loss_d_mu(samples, centroids, alpha=1.0):
    num_samples = samples.size()[0]
    num_classes = centroids.size()[0]
    num_features = samples.size()[1]

    samples = samples.double()
    centroids = centroids.double()

    q = calc_soft_assignment(samples, centroids, alpha=alpha)
    q = q.cuda()
    p = calc_auxiliary_target_distribution(q)
    p = p.cuda()

    # for calculating derivative
    p_min_q = p - q  # num_samples * num_classes
    p_min_q = p_min_q.cuda()

    d_loss_d_mu = torch.zeros((num_classes, num_features), dtype=torch.double).cuda()
    for idx_k in range(num_classes):
        tmp_sigma = torch.zeros((1, num_features), dtype=torch.double).cuda()
        for idx_sample in range(num_samples):
            tmp_term_1 = (
                1.0
                + torch.sum((samples[idx_sample, :] - centroids[idx_k, :]) ** 2) / alpha
            ) ** (-1.0)
            tmp_term_2 = p[idx_sample, idx_k] - q[idx_sample, idx_k]
            tmp_term_3 = samples[idx_sample, :] - centroids[idx_k, :]
            tmp_sigma += (tmp_term_1 * tmp_term_2 * tmp_term_3).view(1, -1)

        d_loss_d_mu[idx_k] = (-((alpha + 1.0) / alpha) * tmp_sigma).cuda()

    labels = get_clustering_labels(q)

    return d_loss_d_mu, labels


def get_clustering_labels(mat_soft_assignment):
    """

    :param mat_soft_assignment: num_samles * num_features
    :return: labels
    """

    soft_assignment_np = mat_soft_assignment.cpu().detach().numpy().copy()

    labels = np.argmax(soft_assignment_np, axis=1)

    return labels


def calc_t_dstr_from_dstn_mat(dstn_mat):
    num_samples = dstn_mat.shape[0]
    coef_power = -1

    xx, yy = np.meshgrid(np.arange(num_samples), np.arange(num_samples))
    numerater_vector = (1.0 + dstn_mat**2) ** coef_power

    denominater_vector = torch.sum(numerater_vector)
    # denominater_vector = torch.sum(numerater_vector) - numerater_vector
    ret_vector = numerater_vector / denominater_vector

    return ret_vector


def calc_t_dstr_from_samples(samples, dstn_max_value=np.inf):
    """
    calculate t distribution.
    https://jp.mathworks.com/help/stats/t-sne.html#bvkwu5p
    :param dstn_max_value:
    :param samples: num_samples * num_features
    :return: t distribution. num_samples * num_samples
    """

    num_samples = samples.shape[0]
    # num_features = samples.shape[1]
    coef_power = -1.0

    #     calculate numerater
    # numerater = torch.zeros((num_samples, num_samples)).cuda()
    # ret = torch.zeros((num_samples, num_samples)).cuda()
    xx, yy = np.meshgrid(np.arange(num_samples), np.arange(num_samples))
    numerater_vector = (
        1.0 + torch.sum((samples[xx] - samples[yy]) ** 2, dim=2)
    ) ** coef_power

    min_value = (1.0 + dstn_max_value**2) ** coef_power
    numerater_vector = torch.clamp(numerater_vector, min_value, 1.0)

    # distance matrix ver
    # dstn_mat = calc_dstn_mat(samples, dstn_max_value=dstn_max_value)
    # numerater_vector_dstn_mat = (1.0 + dstn_mat**2.0) ** coef_power

    # set diag elements to zero, based on the definition
    idx_diag = np.zeros(num_samples)
    for i_idx_diag in range(num_samples):
        idx_diag[i_idx_diag] = i_idx_diag + i_idx_diag * num_samples
    numerater_vector.view(-1)[idx_diag] = 0

    denominater_vector = torch.sum(numerater_vector)
    # denominater_vector = torch.sum(numerater_vector) - numerater_vector
    ret_vector = numerater_vector / denominater_vector

    return ret_vector


def calc_dstn_mat(samples, metric="euclid", dstn_max_value=np.inf):
    # https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    samples_norm = (samples**2).sum(1).view(-1, 1)
    samples_t = torch.transpose(samples, 0, 1)
    samples_t_norm = samples_norm.view(1, -1)

    dist = torch.clamp(
        samples_norm + samples_t_norm - 2.0 * torch.mm(samples, samples_t), 0.0, np.inf
    )  # clamp is necessary before sqrt for managing very small negative value
    dist = torch.sqrt(dist)

    # replace larger values than 'distance_max_value'
    return torch.clamp(dist, 0.0, dstn_max_value)


def calc_kld_t_dstr(p, q, normalize=True):
    assert p.shape == q.shape

    #     if i == j (i.e. dialog elements), the element would not be included in the summation
    # sum of triu element
    triu = np.triu(np.ones(p.shape, dtype=np.int64), k=1)
    idx_triu = np.where(triu.flatten() == 1)[0]

    kld = torch.sum(
        p.view(-1)[idx_triu] * torch.log(p.view(-1)[idx_triu] / q.view(-1)[idx_triu])
    )

    if normalize:
        # divide by number of nonzero element
        kld = kld / (len(idx_triu))

    # TODO for debug
    if kld < 0:
        print()

    return kld


def calc_kld_sparse(latents, p):
    """

    :param latents: num_samples * num_features torch tensor
    :param p: float value
    :return:
    """

    #     latents should be 0 - 1
    latents_abs = torch.abs(latents)
    p_hat = torch.clamp(latents_abs, min=0, max=1)

    kld = torch.mean(
        p * torch.log(p / p_hat) + (1 - p) * torch.log((1 - p) / (1 - p_hat))
    )

    return kld
