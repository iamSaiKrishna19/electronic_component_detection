Inventory Detection & Management System Using YOLOv5
Project Overview

This project is a real-time object detection and inventory management system for electronic components. Using a webcam and YOLOv5, it detects components like Arduino Nano, DHT11, Ultrasonic Sensor, Breadboard, and LCD 16x2, counts them, and stores the data in a database (MySQL/PostgreSQL).

It provides a GUI dashboard for displaying, searching, deleting, and exporting inventory data.

Features

Real-time Object Detection: Detects electronic components with YOLOv5 and OpenCV.

Automatic Inventory Counting: Updates component counts in the database automatically.

Database Integration: Supports PostgreSQL and MySQL with safe, parameterized queries.

User Email Tracking: Tracks inventory per user via email.

GUI Interface (Tkinter):

Start new detection session

Search by component name or email

Delete specific entries

Display entire database

Export inventory data to Excel

Safe UPSERT: Automatically inserts or updates quantities if the component already exists.

Tech Stack

Programming Language: Python 3.x

Computer Vision: OpenCV, YOLOv5

GUI: Tkinter, PIL

Database: PostgreSQL / MySQL

Other Libraries: pandas, psycopg2 (PostgreSQL), mysql-connector-python (MySQL), tabulate, PyYAML