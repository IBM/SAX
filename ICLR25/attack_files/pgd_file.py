import torch
import torch.nn as nn

from attack_files.attack_file import Attack

class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=8/255,
                 alpha=2/255, steps=10, random_start=True, 
                 builtin_exp=False):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.builtin_exp = builtin_exp

    def forward(self, images, labels, reconstruction, threshold):
        r"""
        Overridden.
        """
        self._check_inputs(images)

        mask = reconstruction.ge(threshold)
        mask2 = reconstruction.lt(threshold)

        mask = mask.view(images.shape[0], images.shape[2], images.shape[3]).unsqueeze(1)
        mask2 = mask2.view(images.shape[0], images.shape[2], images.shape[3]).unsqueeze(1)

        if images.shape[1]==3:
            mask = mask.repeat(1, 3, 1, 1)
            mask2 = mask2.repeat(1, 3, 1, 1)


        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            elif self.builtin_exp:
                cost = loss(outputs, labels)
            else:
                cost = loss(outputs[0], labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            old_adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            old_adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            adv_images = ((images.detach() * mask.detach()) + (old_adv_images.detach() * mask2.detach())).detach()

        return adv_images
