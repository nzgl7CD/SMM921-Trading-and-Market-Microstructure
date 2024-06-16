def calculate_covariance_matrix(self, window_returns):
  # Calculate the sample covariance matrix
  covariance_matrix = window_returns.cov()
  
  # Replace all cross-asset correlations with the average cross-asset correlation
  avg_correlation = (covariance_matrix.values - np.diag(np.diag(covariance_matrix.values))).mean()
  adjusted_covariance_matrix = avg_correlation + np.diag(np.diag(covariance_matrix.values))
  
  return pd.DataFrame(adjusted_covariance_matrix, index=covariance_matrix.index, columns=covariance_matrix.columns)

def mean_variance_optimal_weights(self, alphas, cov_matrix, risk_aversion):
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        ones = np.ones(len(alphas))

        # Calculate theta_t
        numerator = ones.T @ inv_cov_matrix @ alphas
        denominator = ones.T @ inv_cov_matrix @ ones
        theta = numerator / denominator

        # Calculate optimal weights
        weights = (1 / risk_aversion) * inv_cov_matrix @ (alphas - ones * theta)
        return weights

    def calculate_portfolio_returns_and_weights(self, crra=4):
        portfolio_returns = []
        portfolio_weights = []
        for i in range(60, len(self.returns) - 1):
            window_returns = self.returns.iloc[i-60:i, 1:-1]
            cov_matrix = self.calculate_covariance_matrix(window_returns)
            alphas = self.alphas.iloc[i-60, 1:].values
            weights = self.mean_variance_optimal_weights(alphas, cov_matrix, crra)
            next_month_return = self.returns.iloc[i+1, 1:-1].dot(weights)
            portfolio_returns.append(next_month_return)
            portfolio_weights.append(weights)
        self.portfolio_returns = pd.Series(portfolio_returns)
        self.portfolio_weights = pd.DataFrame(portfolio_weights)
        return f'Returns:\n\n{self.portfolio_returns}\n\nWeights:\n{self.portfolio_weights}'
