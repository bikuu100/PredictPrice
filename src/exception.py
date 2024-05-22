# exception.py

class CustomError(Exception):
    """Base class for other exceptions"""
    pass

class InputError(CustomError):
    """Raised when there is an input error"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class CalculationError(CustomError):
    """Raised when there is an error in a calculation"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
