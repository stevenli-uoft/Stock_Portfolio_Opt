from src.data.database_setup import DatabaseSetup


def test_database_setup():
    # Create an instance of DataCollection
    db_setup = DatabaseSetup()

    # Connect to the database
    if db_setup.sql_connect():
        print("Connected to the database successfully.")

        # Create StockData table
        if db_setup.create_stock_data_table():
            print("StockData table created successfully.")

        # Disconnect from the database
        db_setup.sql_disconnect()
    else:
        print("Failed to connect to the database.")


if __name__ == "__main__":
    test_database_setup()
