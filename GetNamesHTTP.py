import urllib2
import os

fileName = "C:\\Users\\TeaganCole\\Downloads\\15-17\\2016"
files = os.listdir(fileName)

out = open("2017.txt", 'w')

for f in files:
	names  = f.split("_")
	
	base = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK="
	urlBase = base + names[0] + '&type=10-K'
	response = urllib2.urlopen(urlBase)
	html = response.read()
	indexOfStart = html.find('<span class="companyName">') #26
	indexOfStart += 26
	indexOfEnd = html.find('<', indexOfStart)
	stockName = html[indexOfStart:indexOfEnd]
	indexOf10K = html.find('10-K')
	indexOfTD = html.find('<td>', indexOf10K)
	indexOfYear = html.find('2017', indexOfTD)
	if indexOfYear == -1:
		indexOf10K = html.find('10-K', indexOfTD)
		indexOfTD = html.find('<td>', indexOf10K)
		indexOfYear = html.find('2017', indexOfTD)
		if indexOfYear == -1:
			indexOf10K = html.find('10-K', indexOfTD)
			indexOfTD = html.find('<td>', indexOf10K)
			indexOfYear = html.find('2017', indexOfTD)
	if indexOfYear == -1:
		year = 'Not Found'
	else:
		year = html[indexOfYear: indexOfYear + 10]
	
	
	
	
	out.write(names[0] + '\t' + stockName + '\t' + year + '\r')
	
	
		