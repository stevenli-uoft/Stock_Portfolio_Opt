import mysql.connector
from src.config import DB_CONFIG


class DatabaseSetup:
    """A class that gathers historical data of the stocks of a portfolio.
    Each stock will have a table in the database with columns adjusted_data,
    volume, and returns.
    """

    def __init__(self):
        self.connection = None

    def sql_connect(self) -> bool:
        """Establish a connection to the database <dbname> using the
        username <username> and password <password>, and assign it to the
        instance attribute <connection>.

        Return True if the connection was made successfully, False otherwise.
        I.e., do NOT throw an error if making the connection fails.
        """
        try:
            self.connection = mysql.connector.connect(
                host=DB_CONFIG['host'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                database=DB_CONFIG['database']
            )
            return True
        except mysql.connector.Error:
            return False

    def sql_disconnect(self) -> bool:
        """Close this connection to the database.

        Return True if closing the connection was successful, False otherwise.
        I.e., do NOT throw an error if closing the connection failed.
        """
        try:
            if self.connection:
                self.connection.close()
            return True
        except mysql.connector.Error:
            return False

    def create_stock_data_table(self):
        """Create the StockData table in the database."""
        cursor = self.connection.cursor()
        try:
            cursor.execute("DROP TABLE IF EXISTS StockData")
            cursor.execute("""
                CREATE TABLE StockData (
                    Ticker VARCHAR(15),
                    Date DATE,
                    AdjClose FLOAT,
                    Volume BIGINT,
                    PRIMARY KEY (Ticker, Date)
                )
            """)
        except mysql.connector.Error as e:
            print(f"Error: {e}")
            self.connection.rollback()
        finally:
            cursor.close()

    def reset_database(self) -> None:
        """Function called in main if the user decides to clear the database
        and insert a new or re-run a portfolio with the model"""
        cursor = self.connection.cursor()
        delete_query = """DROP SCHEMA IF EXISTS optimization_data """
        create_query = """CREATE SCHEMA IF NOT EXISTS optimization_data"""
        try:
            cursor.execute(delete_query)
            cursor.execute(create_query)
            cursor.close()
        except mysql.connector.Error:
            self.connection.rollback()
            cursor.close()
