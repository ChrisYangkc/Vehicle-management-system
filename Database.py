import os
import pickle
import random

class Database:
    # Represents a simple database for storing and retrieving data in a file using pickle.
    def __init__(self, filename="vehicle_status.data"):
        # Initialize the database with a specified filename or a default name.
        self.filename = filename

    def file_exists(self):
        # Check if the data file exists.
        return os.path.exists(self.filename)

    def create_file(self):
        # Create a new data file or overwrite an existing one with an empty list.
        with open(self.filename, 'wb') as file:
            pickle.dump([], file)

    def write_to_file(self, data):
        # Store the given data in the data file.
        with open(self.filename, 'wb') as file:
            pickle.dump(data, file)

    def read_from_file(self):
        # Read and return the data from the data file. 
        # If the file doesn't exist, it gets created.
        if not self.file_exists():
            self.create_file()
        with open(self.filename, 'rb') as file:
            return pickle.load(file)

    def clear_file(self):
        # Clear the contents of the data file by overwriting it with an empty list.
        self.create_file()