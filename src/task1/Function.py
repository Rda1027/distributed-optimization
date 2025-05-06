class Function:
    def __init__(self, fn, fn_grad):
        self.fn = fn
        self.fn_grad = fn_grad

    def __call__(self, x):
        return self.fn(x)

    def grad(self, x):
        return self.fn_grad(x)