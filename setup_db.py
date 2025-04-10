import mysql.connector
from mysql.connector import Error
from passlib.hash import sha256_crypt

def setup_database():
    try:
        # Connect to MySQL server
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='Dhanu777%'
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            
            # Create database
            cursor.execute("CREATE DATABASE IF NOT EXISTS cybergaruna")
            print("Database 'cybergaruna' created successfully")
            
            # Switch to the database
            cursor.execute("USE cybergaruna")
            
            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL
                )
            """)
            print("Table 'users' created successfully")
            
            # Create regcases table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS regcases (
                    id BIGINT PRIMARY KEY,
                    ctype VARCHAR(100) NOT NULL,
                    cdescription TEXT NOT NULL,
                    clink TEXT NOT NULL
                )
            """)
            print("Table 'regcases' created successfully")
            
            # Create acceptedcomplaints table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS acceptedcomplaints (
                    id BIGINT PRIMARY KEY,
                    ctype VARCHAR(100) NOT NULL,
                    cdescription TEXT NOT NULL,
                    clink TEXT NOT NULL,
                    officer VARCHAR(100) NOT NULL
                )
            """)
            print("Table 'acceptedcomplaints' created successfully")
            
            # Create rejectedcomplaints table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rejectedcomplaints (
                    id BIGINT PRIMARY KEY,
                    ctype VARCHAR(100) NOT NULL,
                    cdescription TEXT NOT NULL,
                    clink TEXT NOT NULL,
                    officer VARCHAR(100) NOT NULL
                )
            """)
            print("Table 'rejectedcomplaints' created successfully")
            
            # Clear existing users
            cursor.execute("DELETE FROM users")
            
            # Create a new password hash using sha256_crypt
            password = "admin123"
            password_hash = sha256_crypt.hash(password)
            
            # Insert default admin user with new hash
            cursor.execute("""
                INSERT INTO users (username, password) 
                VALUES (%s, %s)
            """, ('admin', password_hash))
            connection.commit()
            print("Default admin user created successfully")
            print("Username: admin")
            print("Password: admin123")
            
    except Error as e:
        print(f"Error: {e}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection closed")

if __name__ == "__main__":
    setup_database() 