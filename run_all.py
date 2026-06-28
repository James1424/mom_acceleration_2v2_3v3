from src.get_holdings_universe import main as build_universe
from src.download_data import main as download_prices
from src.backtest_acceleration import main as backtest
from src.update_readme import main as update_readme

if __name__ == "__main__":
    build_universe()
    download_prices()
    backtest()
    update_readme()
