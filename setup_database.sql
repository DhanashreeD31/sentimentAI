-- Create the database
CREATE DATABASE IF NOT EXISTS cybergaruna;
USE cybergaruna;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
);

-- Create regcases table (for registered complaints)
CREATE TABLE IF NOT EXISTS regcases (
    id BIGINT PRIMARY KEY,
    ctype VARCHAR(100) NOT NULL,
    cdescription TEXT NOT NULL,
    clink TEXT NOT NULL
);

-- Create acceptedcomplaints table
CREATE TABLE IF NOT EXISTS acceptedcomplaints (
    id BIGINT PRIMARY KEY,
    ctype VARCHAR(100) NOT NULL,
    cdescription TEXT NOT NULL,
    clink TEXT NOT NULL,
    officer VARCHAR(100) NOT NULL
);

-- Create rejectedcomplaints table
CREATE TABLE IF NOT EXISTS rejectedcomplaints (
    id BIGINT PRIMARY KEY,
    ctype VARCHAR(100) NOT NULL,
    cdescription TEXT NOT NULL,
    clink TEXT NOT NULL,
    officer VARCHAR(100) NOT NULL
);

-- Create a default admin user (password will be "admin123")
INSERT INTO users (username, password) VALUES 
('admin', '$5$rounds=535000$eWPzAJvnsYZb8bWn$hDQzT9KzsYucZYq.vif9BtYh8yaAW1bXc8hBkN9PNC7'); 