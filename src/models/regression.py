
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def get_models():
    return {
        "Linear": LinearRegression(),
        "RF": RandomForestRegressor(),
        "GBM": GradientBoostingRegressor()
    }
