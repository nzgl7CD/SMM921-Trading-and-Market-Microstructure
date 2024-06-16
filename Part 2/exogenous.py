class Exogenous:
    def __init__(self) -> None:
        self.risk_free_rate=0.02
        self.ic=0.02
        self.annual_factor=12
        self.crra=4
    def get_risk_free_rate(self):
        return self.risk_free_rate
    def get_ic(self):
        return self.ic
    def get_annual_factor(self):
        return self.annual_factor
    def get_crra(self):
        return self.crra
    