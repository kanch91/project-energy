import datetime
import pandas
import csv
import time
import tabula

start_time = time.time() # For monitoring the execution time

start_date = datetime.datetime.strptime("01-01-2017", "%d-%m-%Y") # Start date of hourly load data
end_date = datetime.datetime.strptime("26-11-2020", "%d-%m-%Y") # End date of Hourly load data

# List of all the dates to scrape the hourly load data
generated_date = [start_date + datetime.timedelta(days=x) for x in range(0, (end_date-start_date).days)]

# Url for scraping the load data for every hour
url = "https://mahasldc.in/wp-content/reports/dr0_"

# Creating 'loaddata.csv' to store the data
with open('hourly_loaddata.csv', mode='w') as csv_file:

  """ Format of the CSV to store the load data (hourly_loaddata.csv)- 
  Date: DMMYYYY or DDMMYYYY;
  Slot: Each hour divided in slots (0-23);
  Load: Load consumption data in integer;
  Temperature: Hourly temperature in degree celsius;
  Humidity: Hourly humidity in %
  """

  fieldnames = ['Date', 'Slot', 'Load', 'Temperature', 'Humidity'] # Columns of the data file
  writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
  writer.writeheader()

  # Loop for scraping the data for each date
  for date in generated_date:
    try:
      current_url = url + date.strftime("%d%m%Y") + ".pdf" 

      # Using tabula.py to extract data in a table from the PDFs
      tabula.convert_into(current_url, "pdf_data.csv", pages=4) #Jumping to the fourth page where the table for the load data is listed

      # Extracting only the load data of Mumbai from the Table
      prefinal_data = pandas.read_csv('pdf_data.csv')

      # Loop for extracting the data for each hour
      for i in range(24):
        value = int(prefinal_data["Unnamed: 12"][i+4]) # Extracting the value & casting
        writer.writerow({'Date': date.strftime("%d%m%Y"), 'Slot': i, 'Load': value, 'Temperature': 0, 'Humidity':0})

    except Exception as e:
      print('Exception', e)
      print(date)
      print(current_url)

print(pandas.read_csv('hourly_loaddata.csv'))

# Execution time
end_time = time.time()
print(end_time - start_time, "seconds")


# Weather Data
# import requests
# import csv
# import os
# from bs4 import BeautifulSoup
# url = 'https://www.wunderground.com/history/daily/in/mumbai/VABB/date/'
#
# date = "2020-11-16"
#
# current_url = url + date
# resp = requests.get(current_url)
# soup = BeautifulSoup(resp.text, 'lxml')
# main_div = soup.find_all('div')
#
# import requests
# import csv
# import os
# from bs4 import BeautifulSoup
# url = 'https://www.wunderground.com/history/daily/in/mumbai/VABB/date/'
#
# date = "2020-11-16"
#
# current_url = url + date
# resp = requests.get(current_url)
# soup = BeautifulSoup(resp.text, 'lxml')
# print(soup.prettify())
#