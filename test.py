#Library
import urllib3

#API
http = urllib3.PoolManager()
response = http.request('GET', 'https://eservices.mas.gov.sg/apimg-gw/server/monthly_statistical_bulletin_non610mssql/domestic_interest_rates_daily/views/domestic_interest_rates_daily', headers={'keyid': 'XXX'})

print(response.data)
print(response.data.decode('utf-8'))
print(response.status)
print(response.headers['Content-Type'])

