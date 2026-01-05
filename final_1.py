import yolov5
import cv2
import numpy as np
import yaml
import tkinter as tk
from PIL import Image, ImageTk
import time
import mysql.connector
import pandas as pd
import subprocess
from tabulate import tabulate
from tkinter import ttk
from tkinter import simpledialog
import tkinter.messagebox

# MySQL DB connection parameters
DB_PARAMS = {
    'database': 'electronic',
    'user': 'root',
    'password': '1234',
    'host': 'localhost',
    'port': 3306
}

def create_connection():
    return mysql.connector.connect(**DB_PARAMS)

def readSQL():
    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM my_table")
        rd = cursor.fetchall()
        data = {}
        for row in rd:
            data[row[1]] = row[2]
        return data
    except Exception:
        return {}
    finally:
        if conn.is_connected():
            conn.close()

def writeToSQL(data):
    global user_email, label_counts
    t_data = [(c[0], c[1], user_email) for c in data]

    if readSQL() == label_counts:
        return data

    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS my_table (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            qty INT,
            email VARCHAR(255),
            UNIQUE KEY unique_user_component (email, name)
        )
    """)

    for row in t_data:
        cursor.execute("""
            INSERT INTO my_table (name, qty, email)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE qty = qty + VALUES(qty)
        """, row)

    conn.commit()
    conn.close()
    return data

def convert_to_excel():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()

    for table in tables:
        table_name = table[0]
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        name = simpledialog.askstring("Excel Sheet Name", "Enter Excel file name:")
        excel_file = f"{name}.xlsx"
        df.to_excel(excel_file, index=False)

    conn.close()
    subprocess.Popen(["start", excel_file], shell=True)

def display_data():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM my_table")
    data = cursor.fetchall()
    conn.close()

    if data:
        headers = ['ID', 'Name', 'Quantity', 'Email']
        window = tk.Tk()
        window.title("MySQL Data")
        table = ttk.Treeview(window, columns=headers, show='headings')
        for h in headers:
            table.heading(h, text=h)
        for row in data:
            table.insert('', 'end', values=row)
        table.pack()
        window.mainloop()

def search_by_email():
    email = simpledialog.askstring("Search", "Enter Email:")
    if not email:
        return
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM my_table WHERE email=%s", (email,))
    data = cursor.fetchall()
    conn.close()

    if data:
        window = tk.Tk()
        headers = ['ID', 'Name', 'Quantity', 'Email']
        table = ttk.Treeview(window, columns=headers, show='headings')
        for h in headers:
            table.heading(h, text=h)
        for row in data:
            table.insert('', 'end', values=row)
        table.pack()
        window.mainloop()
    else:
        tk.messagebox.showinfo("No Data", "No records found")

def delete_by_email():
    email = simpledialog.askstring("Delete", "Enter Email:")
    name = simpledialog.askstring("Delete", "Enter Component Name:")
    if not email or not name:
        return
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM my_table WHERE email=%s AND name=%s", (email, name))
    conn.commit()
    conn.close()
    tk.messagebox.showinfo("Deleted", "Record deleted (if existed)")

def show_entire_database():
    display_data()

def search_by_name():
    name = simpledialog.askstring("Search", "Enter Component Name:")
    if not name:
        return
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM my_table WHERE name=%s", (name,))
    data = cursor.fetchall()
    conn.close()

    if data:
        window = tk.Tk()
        headers = ['ID', 'Name', 'Quantity', 'Email']
        table = ttk.Treeview(window, columns=headers, show='headings')
        for h in headers:
            table.heading(h, text=h)
        for row in data:
            table.insert('', 'end', values=row)
        table.pack()
        window.mainloop()
    else:
        tk.messagebox.showinfo("No Data", "No records found")

def main_menu():
    root = tk.Tk()
    root.title("Menu")
    root.geometry("300x300")

    tk.Button(root, text="New Detection", command=start_detection_session).pack(pady=5)
    tk.Button(root, text="Search by Name", command=search_by_name).pack(pady=5)
    tk.Button(root, text="Search by Email", command=search_by_email).pack(pady=5)
    tk.Button(root, text="Delete Entry", command=delete_by_email).pack(pady=5)
    tk.Button(root, text="Show Database", command=show_entire_database).pack(pady=5)

    root.mainloop()

def start_detection_session():
    global user_email
    user_email = simpledialog.askstring("Email", "Enter Email:")
    if user_email:
        start_detection_gui()

def start_detection_gui():
    global label_counts, freeze_time

    window = tk.Tk()
    window.title("YOLO Detection")

    image_label = tk.Label(window)
    image_label.pack()

    counts_label = tk.Label(window)
    counts_label.pack()

    cap = cv2.VideoCapture(0)
    label_counts = {}
    freeze_time = None
    freeze_duration = 3

    def update_frame():
        nonlocal cap
        ret, frame = cap.read()
        if not ret:
            return

        results = model(frame)
        preds = results.pred[0].cpu().numpy()
        new = False

        for p in preds:
            cls = labels[int(p[5])]
            label_counts[cls] = label_counts.get(cls, 0) + 1
            new = True

        if new:
            writeToSQL([[k, v] for k, v in label_counts.items()])

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        image_label.config(image=img)
        image_label.image = img

        counts_label.config(text=str(label_counts))
        window.after(10, update_frame)

    update_frame()
    window.mainloop()
    cap.release()

# Load YOLO model
model = yolov5.load('best.pt')
model.conf = 0.8
model.iou = 0.95
input_size = (640, 480)

with open('ml.yaml', 'r') as f:
    labels = yaml.safe_load(f)['names']

if __name__ == "__main__":
    main_menu()
