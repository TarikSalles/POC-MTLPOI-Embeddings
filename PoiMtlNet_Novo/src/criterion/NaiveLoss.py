class NaiveLoss:
    def __init__(self, alpha=0.5, beta=0.5, min_weight=1e-4, max_weight=10.0):
        self.alpha = alpha
        self.beta = beta
        self.min_weight = min_weight
        self.max_weight = max_weight

    def compute_loss(self, loss_next, loss_category, adjustment_factor=0.01):
        self.adjust_alpha_beta(loss_next, loss_category, adjustment_factor)

        total = self.alpha + self.beta
        norm_alpha = self.alpha / total
        norm_beta = self.beta / total

        return norm_alpha * loss_next + norm_beta * loss_category

    def backward(self, loss_next, loss_category):
        loss = self.compute_loss(loss_next, loss_category)
        loss.backward()
        return loss

    def compute_loss_no_adjustment(self, loss_next, loss_category):
        total = self.alpha + self.beta
        norm_alpha = self.alpha / total
        norm_beta = self.beta / total

        return norm_alpha * loss_next + norm_beta * loss_category

    def adjust_alpha_beta(self, loss_next, loss_category, adjustment_factor=0.01):
        norm_loss_next = loss_next.item()
        norm_loss_category = loss_category.item()

        if norm_loss_category < norm_loss_next:
            self.alpha *= (1 + adjustment_factor)
            self.beta *= (1 - adjustment_factor)
        elif norm_loss_next < norm_loss_category:
            self.alpha *= (1 - adjustment_factor)
            self.beta *= (1 + adjustment_factor)

        self.alpha = max(min(self.alpha, self.max_weight), self.min_weight)
        self.beta = max(min(self.beta, self.max_weight), self.min_weight)
