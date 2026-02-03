"""
File: user_model.py
Description: User data model for handling user entities and authentication.
Dependencies: None (can be extended with SQLAlchemy, Pydantic, etc.)
Author: Sample Team

This is a sample file demonstrating proper code structure and documentation.
Students should replace this with their actual implementation.
"""


class User:
    """
    Represents a user entity in the system.
    
    Attributes:
        user_id (str): Unique identifier for the user.
        username (str): The user's username.
        email (str): The user's email address.
        role (str): The user's role (e.g., 'admin', 'merchant_user', 'consumer').
    
    Methods:
        validate_email(): Validates the email format.
        to_dict(): Converts the user object to a dictionary.
    """
    
    def __init__(self, user_id: str, username: str, email: str, role: str = 'consumer'):
        """
        Initialize a new User instance.
        
        Args:
            user_id (str): Unique identifier for the user.
            username (str): The user's username.
            email (str): The user's email address.
            role (str, optional): The user's role. Defaults to 'consumer'.
        
        Raises:
            ValueError: If any required field is empty.
        """
        if not all([user_id, username, email]):
            raise ValueError("User ID, username, and email are required")
        
        self.user_id = user_id
        self.username = username
        self.email = email
        self.role = role
    
    def validate_email(self) -> bool:
        """
        Validates the email format.
        
        Returns:
            bool: True if email is valid, False otherwise.
        """
        # Simple validation - students should implement proper validation
        return '@' in self.email and '.' in self.email.split('@')[1]
    
    def to_dict(self) -> dict:
        """
        Converts the user object to a dictionary.
        
        Returns:
            dict: Dictionary representation of the user.
        """
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'role': self.role
        }
