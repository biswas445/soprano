start_soprano.bat file from root is the main cli which opens server.py file. server.py file is an interactive menu. it provides option for user such as:

option 1: API
option 2: Test_API "it launches both api.py and test_api.py  files inside of the server in 10sec delay. after the sucessful run it save the file inside of the server folder.
"note: api does not save any file it receive request and send the file using http request, test file just grab the incoming file from http"
option: 3: websocket
option 4: same as option 2 but websocket send the files into chucks test file only play the chunk. 
option 5: lauch webui from soprano
option 6: lauches soprano cli.
option 7: exit.

how this start_soprano.bat file works:
for each option it open new command terminal and run the each file and close itslef but not options that it lauch. 
