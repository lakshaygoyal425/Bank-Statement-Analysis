import os
import re
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pytesseract

from pdf2image import convert_from_path
from PIL import Image

import camelot
import tabula

pytesseract.pytesseract.tesseract_cmd = r'E:\bank-statement-analysis-master\venv\Lib\site-packages\Tesseract-OCR\tesseract.exe'


def read_image(image):
    fulltext = pytesseract.image_to_string(image, lang='eng')
    return fulltext


# Takes a pdf path and return list of images path
def pdf_to_images(pdf_path):
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    images = convert_from_path(pdf_path, 200)
    return np.array(images[0])


def classify_bank(text):
    banks = {"YES": "YES BANK"}
    ifsc = get_ifsc(text)
    bank = ""
    for j in banks.keys():
        if j in ifsc:
            bank = banks[j]
            break

    return ifsc, bank


def get_ifsc(text):
    def replace(text):
        return text.replace('?', '7')

    ifsc = text.find('IFSC')
    new_text = text[ifsc: ifsc + 30]
    new_text = replace(new_text)

    code = re.findall(r'[A-Z0-9]{11}', new_text)[0]

    return code


def get_acc(text):
    if '-' in list(text):
        text = text.replace('-', '')

    index = text.lower().find('account n')
    try:
        text = re.findall(r'[0-9]{9,18}', text[index:])[0]
    except:
        return 0
    return text


def get_name(info):
    title = ["mr.", "shri", "ms.", "mrs."]
    for i in info:
        for j in title:
            if j in i.lower():
                return i.lower().replace(j, "").upper()
    return -1


def month_diff(d1, d2):
    return abs(d1.month - d2.month + 12 * (d1.year - d2.year))


def extract_data(pdf_path):
    im = pdf_to_images(pdf_path)
    h, w, _ = im.shape

    crop = im[1:h // 3, :, :]
    # OCR
    info = read_image(crop)
    # Indentify bank
    ifsc, bank = classify_bank(info)
    print(ifsc, bank)
    # Get account name
    name = get_name(info.split("\n"))
    # Get account number
    acc_no = get_acc(info)

    print("Information:")
    print(acc_no)
    print("IFSC = ", ifsc)
    print("Bank = ", bank)
    print("Name = ", name)

    print("Exracting transactions...")
    if bank == "YES BANK":
        yes_bank(pdf_path)
    else:
        print("Not available")

    return name, acc_no, bank, ifsc


def yes_bank(pdf_path):
    page = 2

    df = tabula.read_pdf(pdf_path, pages="1")[0]

    while True:
        p = tabula.read_pdf(pdf_path, pages=str(page))[1]
        if "Unnamed: 0" in p.columns:
            p = p.drop(["Unnamed: 0"], axis=1)
        # print(p.columns)
        if "Description" in p.columns:
            df = pd.concat([df, p], axis=0)
        else:
            break
        page += 1
    df.index = list(range(0, len(df)))

    for i in df.index:
        if type(df["Transaction\rDate"][i]) == float:
            df["Transaction\rDate"][i] = df["Transaction"][i]
    df["Transaction Date"] = df["Transaction\rDate"]
    df = df.drop(["Transaction", "Transaction\rDate"], axis=1)

    delete = []
    headers = ["Date", "Description", "Credit", "Debit", "Balance"]

    for i in df.index:
        row = df.iloc[i, :].tolist()
        nan_c = 0
        # For checking empty rows
        for j in row:
            try:
                if np.isnan(j):
                    nan_c += 1
            except:
                continue
        if nan_c == len(df.columns):
            delete.append(i)

        for j in headers:
            if j in row:
                delete.append(i)
    df = df.drop(delete, axis=0)

    last = 0
    delete = []
    for i in df.index:
        if type(df["Value Date"][i]) == float and type(df["Description"][i]) == str:
            buff = df["Description"][last] + df["Description"][i]
            df["Description"][last] = buff
            delete.append(i)
        else:
            last = i
    df = df.drop(delete, axis=0)

    df["Credit"] = df.Credit.apply(lambda x: str(x).replace(",", ""))
    df["Debit"] = df.Debit.apply(lambda x: x.replace(",", ""))

    df["Credit"] = df["Credit"].astype("float64")
    df["Debit"] = df["Debit"].astype("float64")
    df["Value Date"] = df["Value Date"].apply(lambda x: x[3:])

    df = df[["Transaction Date", "Value Date", "Description", "Debit", "Credit", "Balance"]]
    # print(df.head())
    df.to_excel(pdf_path[:pdf_path.find(".")] + ".xlsx", index=False)
