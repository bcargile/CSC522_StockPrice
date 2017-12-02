infile = open("ticker_symbol.txt", 'r')

contents = infile.read()
lines = contents.split('\n')
out = open("spreadsheetready.csv", 'w')
x=1
for l in lines:
	data = l.split('\t')
	if len(data) > 2:
		
		c = 'B' + str(x)
		d = 'C' + str(x)
			
		out.write(data[0] + "\t" + data[1]+ "\t" + data[2] + "\t" + '=GOOGLEFINANCE(' + c + ',"open",'+ d +')' + '\n\n')
		x = x + 2