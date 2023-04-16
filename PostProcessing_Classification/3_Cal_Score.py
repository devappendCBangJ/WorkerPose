import openpyxl
from openpyxl import Workbook
from random import *

compare_number = int(input())

wb = openpyxl.load_workbook(filename = 'score_1.xlsx')
ws = wb.active

for row in ws.iter_rows(1):
    if int(row[0].value) == compare_number:
        print(row[1].value)