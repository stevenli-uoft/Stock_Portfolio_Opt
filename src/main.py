from src.data.database_setup import DatabaseSetup
from src.data.yfinance_handler import YFinanceHandler


def main():
    # Initialize DatabaseSetup and establish a connection
    db_setup = DatabaseSetup()
    if db_setup.sql_connect():
        print("Database connection established.")

        # Create necessary tables
        db_setup.create_stock_data_table()

        # Test disconnection
        if db_setup.sql_disconnect():
            print("Database connection closed.")

    # Initialize YFinanceHandler
    # Assuming you have a user_portfolio.csv file with the portfolio data
    yfinance_handler = YFinanceHandler('data/sample_data')

    # Fetch and process stock data
    yfinance_handler.run()


if __name__ == "__main__":
    main()
