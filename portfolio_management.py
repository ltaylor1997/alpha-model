# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 23:25:16 2022
Currently in the middle of cleaning  up this code. I was very naive on program 
structure and variable naming schemes when this was made. 
It is incredibly poorly written and needs a lot of work. Will be updated over 
coming weeks to reflect my new knowledge I have gained since fall of 2021
@author: lucas
"""
### READ ABOVE PARAGRAPH ### 
import numpy as np
from sklearn import model_selection
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import requests
from fmp_python.fmp import FMP
from datetime import timedelta
from datetime import datetime
from pandas import json_normalize
import json
import requests
from scipy.optimize import minimize
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection

fmp = FMP(api_key='2e1211f60d33c4daa1afec46de6d74f4')

#this will pull be used to pull financial statements
def tickers():
#html for tradeable tickers
    html1 = "https://financialmodelingprep.com/api/v3/available-traded/list?apikey=2e1211f60d33c4daa1afec46de6d74f4"
    #request the html
    x1 = requests.get(html1)
    #convert to json
    json1 = x1.json()
    #convert to df
    tradeable = json_normalize(json1)
    return tradeable
#repeat the protocol with total stock list
def stocks():
    html2 = "https://financialmodelingprep.com/api/v3/stock/list?apikey=2e1211f60d33c4daa1afec46de6d74f4"
    x2 = requests.get(html2)
    json2 = x2.json()
    symbols = json_normalize(json2)
    return symbols
#get the symbols into one dataframe
def get_full_list(tradable, symbols):
    full_symbols = pd.merge(left = tradable, right = symbols,on = ["symbol",'exchangeShortName'])

    #list for initial tickers
    Itickers = []
    #iterate of the rows of the dataframe to get all stocks from NYSE and NASDAQ
    for index, row in full_symbols.iterrows():
        if row['type'] == 'stock'and row['exchangeShortName'] == 'NYSE' or row['exchangeShortName']== 'NASDAQ':
            Itickers.append(row['symbol'])
    return Itickers
#empty lists to capture prices, ratios, and market caps, and company profiles
Iprices = []
Iratios = []
Icap = []
Iprofile = []

def pull2(i,t,prices,ratios,caps,profiles):
    try: #fmp module that gets price data
        if len(Iprices) != len(Icap) or len(Icap) != len(Iratios) or len(Iratios) != len(Iprofile):
            raise Exception("not equal in length")

        html = "https://financialmodelingprep.com/api/v3/historical-price-full/"+t[i]+"?apikey=2e1211f60d33c4daa1afec46de6d74f4"
    #request the html
        x = requests.get(html)
    #convert to json
        json = x.json()
    #convert to df
        df = pd.DataFrame(json['historical'])

        
        prices.append(df)
    #sleep to prevent too many api calls
    #time.sleep(.2)
        html = "https://financialmodelingprep.com/api/v3/ratios/"+t[i]+"?apikey=2e1211f60d33c4daa1afec46de6d74f4"
    #request the html
        x = requests.get(html)
    #convert to json
        json = x.json()
    #convert to df
        df = json_normalize(json)
    #append to list
        ratios.append(df)
    #add wait period to not overload api above 300 calls per minute
    #time.sleep(.2)
        html = "https://financialmodelingprep.com/api/v3/historical-market-capitalization/"+t[i]+"?apikey=2e1211f60d33c4daa1afec46de6d74f4"
        x = requests.get(html)
        json = x.json()
        df = json_normalize(json)
    #time.sleep(.2)
        caps.append(df)

        html = "https://financialmodelingprep.com/api/v3/profile/"+t[i]+"?apikey=2e1211f60d33c4daa1afec46de6d74f4"
        x = requests.get(html)
        json = x.json()
        df = json_normalize(json)
    #time.sleep(.2)
        profiles.append(df)
        print(str(i+1)+" out of "+str(len(t))+ " datapoint gathered")
        return t,prices,ratios,caps,profiles
    except:
        if len(prices) == len(ratios) and len(ratios) == len(caps) and len(caps) == len(profiles):
            t.pop(len(Iprices))
            return t,prices,ratios,caps,profiles
        elif len(prices) != len(ratios) or len(ratios) != len(caps) or len(caps) != len(profiles):
            cut = min([len(prices),len(ratios),len(caps),len(profiles)])
            prices[:cut]
            ratios[:cut]
            caps[:cut]
            profiles[:cut]
            return t,prices,ratios,caps,profiles

# while loop to make sure the program runs automatically
#termination condition is lenngth of tickers and profiles are the same
def run_and_validation(Itickers):
    while len(Iprofile) != len(Itickers):
        
        #get length of caps for initial i
        i = len(Icap)
        
        #use the function to iterate over the list of tickers and get all stock data
        Itickers,Iprices,Iratios,Icap,Iprofile = pull2(i,Itickers,Iprices,Iratios,Icap,Iprofile)
        
        #minimum length of the lists
        cut = min([len(Iprices),len(Iratios),len(Icap),len(Iprofile)])
        
        #print the progress to make sure it's still running


    #cut the lengths to ensure equal lengths, import for data running and accuracy
        Iprices = Iprices[:cut]
        Iratios = Iratios[:cut]
        Icap = Icap[:cut]
        Iprofile = Iprofile[:cut]

#this is for the initial gathering of data. Soemtimes the time runs out
#cuts the lists to equal lengths to run in the proceding code
    cut = min([len(Iprices),len(Iratios),len(Icap),len(Iprofile)])


    Iprices = Iprices[:cut]
    Iratios = Iratios[:cut]
    Icap = Icap[:cut]
    Iprofile = Iprofile[:cut]

#debugging function
def debug(lists,subLists = True):
    try:
        #list to hold the problem list
        problem  = []
        if subLists == True:
        #iterate over the lists of lists
        #works by checkin if the sublists of lists match each other
            for i in range(len(lists)):
                for x in range(len(lists[i])):
                    if len(lists[0][x]) == len(lists[i][x]):
                        pass
                    else:
                        problem.append(i)
                        problem.append(x)
                        raise Exception
        #iterates over to check if lists match
        elif subLists == False:
            for i in range(len(lists)):
                if len(lists[0]) == len(lists[i]):
                    pass
                else:
                    problem.append(i)
                    raise Exception
        print("all lists match in length")
    except:
        raise Exception ("lists don't match in length, problem sublists are "+ str(problem))

#empty lists to get the good prices, ratios, and market caps. Some data that was pulled
def clean(Itickers):
    ratio2 = []
    month_list = []
    cap_list = []
    profiles = []
    tickers = []
#initial cleaning code

    for i in range(len(Iprices)):
    #print(Icap[i])
        if Icap[i].index.size > 270:
        
            if Iprices[i].index.size>270:
                print(i) 
                if Iratios[i].index.size>1:   
                    if Iprofile[i].index.size > 0:
                        ratio2.append(Iratios[i])
                        month_list.append(Iprices[i])
                        cap_list.append(Icap[i])
                        profiles.append(Iprofile[i])
                        tickers.append(Itickers[i])
        else:
            pass


def ratios_drop(ratio2):
    for x in ratio2:
        try:
            for i in range(len(x)):
                x['date'][i] = x['date'][i].replace('-', ' ')
        except:
            pass

    for x in ratio2:
        try:
            for i in range(len(x)):
                x['date'][i] = datetime.strptime(x['date'][i],'%Y %m %d')
        except:
            pass

    for x in range(len(ratio2)):
        try:
            d1 = ratio2[x]['date'].values

            for i in range((len(d1)-1)):
                if i == 0:
                    if d1[0] - d1[1] < timedelta(days = 340):
                        print(x)
                        ratio2[x].drop(1,inplace = True)
                else:
                    if d1[i] - d1[i+1] < timedelta(days = 340):
                        print(x)
                        ratio2[x].drop(i,inplace = True)
        except:
            pass

#function concatenate all of the prices and market caps
def capNprices(appendList,priceList,capList):
    lengthList = []

    #get the lengths of the market caps and prices
    for i in range(len(priceList)):
        lengthList.append([len(priceList[i]),len(capList[i])])
    for i in range(len(priceList)):

        #cut the prices and caps based on which one is smaller for concatenation
        prices = priceList[i][:(min(lengthList[i]))]
        caps = capList[i][:(min(lengthList[i]))]['marketCap']

        #concatenate them
        con = pd.concat([prices,caps],axis = 1)

        #append it to outside list
        appendList.append(con)
    print(appendList)
    return appendList
#empty list for merging
new_month_list = []

#concatenate prices and market caps
new_month_list = capNprices(new_month_list,month_list,cap_list)
print(new_month_list)

def clean_new_month():
    for x in new_month_list:
        try:
            for i in range(len(x)):
                x['date'][i] = x['date'][i].replace('-', ' ')
        except:
            pass

    for x in new_month_list:
        try:
            for i in range(len(x)):
                x['date'][i] = datetime.strptime(x['date'][i],'%Y %m %d')
        except:
            pass

#set the index to eventually cut by date
    new_month_list = [x.set_index('date') for x in new_month_list]
    return new_month_list
#seperate sector
def get_sector():
    sector = [x['sector'] for x in profiles]
    return sector
#add sector column to ratios
def add_secotr(ratio2, sector):
    for i in range(len(sector)):
        ratio2[i]['sector'] = sector[i][0]

    for x in ratio2:
        x.reset_index(drop = True, inplace = True)

#empty ts for chopped ratios
ra2021 = []
ra2020 = []
ra2019 = []
ra2018 = []
ra2017 = []
ra2016 = []

#split ratios by year
def chop2(i, ratio2):
    r21 = ratio2[i][:1]
    r20 = ratio2[i][1:2]
    r19 = ratio2[i][2:3]
    r18 =ratio2[i][3:4]
    r17 =ratio2[i][4:5]
    r16 = ratio2[i][5:6]
    ra2021.append(r21)
    ra2020.append(r20)
    ra2019.append(r19)
    ra2018.append(r18)
    ra2017.append(r17)
    ra2016.append(r16)

def split_ratios(ratio2)  :  
    for i in range(len(ratio2)):
        chop2(i, ratio2)
#put into list for ease of running
entireRatios = [ra2017,ra2018,ra2019,ra2020,ra2021]

for x in entireRatios:
    for x in x:
        x.reset_index(drop = True,inplace = True)

#empty list of dates
def dateCapture():
    global cutDates
    cutDates = []
#function to get dates

    dates = []        
    for x in range(len(entireRatios[i])):
            #prevents error from happening later on
        if len(entireRatios[i][x]) == 0:
            dates.append([])
        else:
                    #gets the date
            x = entireRatios[i][x]['date'][0]
            dates.append(x)
            cutDates.append(dates)
                    
                    

#empty lists
pr2017 = []
pr2018 = []
pr2019 = []
pr2020 = []
pr2021 = []

new_month_list[0].index[0]

#this cuts the prices by the previously captured dates
popList = [ ]
def chop(index1,appList):
    prices = []
    Plist = []
    for i in range(len(cutDates[index1])):

            if type((cutDates[index1][i])) == datetime:
                try:
                    if index1 == 4:
                        x = new_month_list[i][:cutDates[index1][i]]
                        appList.append(x)
                    else:
                        x = new_month_list[i][cutDates[(index1+1)][i]:cutDates[index1][i]]
                        appList.append(x)
                except:
                    Plist.append(i)
                    pass
            else:
                pass

    return Plist
def clean_clists():
    P1 = chop(4,pr2021)

    P2 = chop(3,pr2020)
    P3 = chop(2,pr2019)
    P4 = chop(1,pr2018)
    P5 = chop(0,pr2017)
    global popList
    popList = [P5,P4,P3,P2,P1]

                 
def tickerTracker():
    global tickersByYear 
    tickersByYear = []
    for i in range(len(entireRatios)):
        t = tickers.copy()
        tickersByYear.append(t)
        
        for i in range(len(entireRatios)):
            if len(popList[i]) == 0:
                pass
            elif len(popList[i]) != 0:
                for x in range(len(popList[i])):
                    entireRatios[i].pop((popList[i][x]))
                    tickersByYear[i].pop((popList[i][x]))


#turn into lists for ease of running
year_ratios = []
year_prices = [pr2017,pr2018,pr2019,pr2020,pr2021]

tickers = []

def filter_lists():
    for x in range(len(entireRatios)):
        g = []
        ts = []
        #if length euqals 0, pass it
        for i in range(len(entireRatios[x])):
            if len(entireRatios[x][i]) == 0:
                pass
     #otherwise append it
            else:
                g.append(entireRatios[x][i])
                ts.append(tickersByYear[x][i])
        year_ratios.append(g)
        tickers.append(ts)

#clean the prices and years to account for a potential lack of records
#in easier language, not every firm has ratios that go back 5 years, so this fixes the problem
#creates empy dataframes that mess up data running
c_year_prices = []
c_ratio = []
newTickers = []
def clean2(index):
    need1 = []
    need2 = []
    need3 = []
    for i in range(len(year_prices[index])):
        if year_ratios[index][i].index.size>0:
            if year_prices[index][i].index.size>1:
                need1.append(year_ratios[index][i])
                need2.append(year_prices[index][i])
                need3.append(tickers[index][i])
        else:
            pass
    c_ratio.append(need1)
    c_year_prices.append(need2)
    newTickers.append(need3)

for i in range(len(year_prices)):
    clean2(i)

#reverse dataframes to make them go in chronological order
c_prices = [[x[::-1] for x in x] for x in c_year_prices]

for x in c_prices:
    for x in x:
        x.reset_index(drop = True, inplace = True)

#empty list for market caps
caps = []
#function to get the year market caps
def Mcap(i):
#sublist for caps
    year = []
    #forlist to get all of the caps
    for a in range(len(c_prices[i])):
        x = c_prices[i][a]['marketCap'][0]
        year.append(x)
    #append to list outside of function
    caps.append(year)
for i in range(len(c_prices)):
    Mcap(i)

for i in range(len(caps[0])):
    if caps[0][i] == 0:
        print(i)

caps2 = [np.array(x) for x in caps]

for x in c_ratio:
    for x in x:
        x.reset_index(drop = True, inplace = True)

hprs = []
#this will calculate hodling period returns, useful for 2 resons
def hpr(i):
    r = []

    #get the sub list in c_prices

    for x in range(len(c_prices[i])):
        #price at time 0
        p0 = c_prices[i][x]['close'][:1].values
        #price at time 1
        p1 = c_prices[i][x]['close'][(len(c_prices[i][x])-1):].values
        #holding period return
        hpr = (p1-p0)/p0
        r.append(hpr)
    hprs.append(r)
for i in range(len(c_prices)):
    hpr(i)

#greater than equals closer
#don't like this code, want to make it more efficent
def sortDate(Rlist,Plist,HPRlist,caplist,tlist):
    ratios = []
    prices = []
    returns = []
    caps = []
    tickers = []
    for i in range(len(Rlist)):
        ratios.append([])
        prices.append([])
        returns.append([])
        caps.append([])
        tickers.append([])
    for i in range(len(Rlist)):
        for x in range(len(Rlist[i])):
            if Rlist[i][x]['date'][0] < datetime(2017,6,30) and Rlist[i][x]['date'][0] > datetime(2016,6,30):
                ratios[0].append(Rlist[i][x])
                prices[0].append(Plist[i][x])
                returns[0].append(HPRlist[i][x])
                caps[0].append(caplist[i][x])
                tickers[0].append(tlist[i][x])
            elif Rlist[i][x]['date'][0] < datetime(2018,6,30) and Rlist[i][x]['date'][0] > datetime(2017,6,30):
                ratios[1].append(Rlist[i][x])
                prices[1].append(Plist[i][x])
                returns[1].append(HPRlist[i][x])
                caps[1].append(caplist[i][x])
                tickers[1].append(tlist[i][x])
            elif Rlist[i][x]['date'][0] < datetime(2019,6,30) and Rlist[i][x]['date'][0] > datetime(2018,6,30):
                ratios[2].append(Rlist[i][x])
                prices[2].append(Plist[i][x])
                returns[2].append(HPRlist[i][x])
                caps[2].append(caplist[i][x])
                tickers[2].append(tlist[i][x])
            elif Rlist[i][x]['date'][0] < datetime(2020,6,30) and Rlist[i][x]['date'][0] > datetime(2019,6,30):
                ratios[3].append(Rlist[i][x])
                prices[3].append(Plist[i][x])
                returns[3].append(HPRlist[i][x])
                caps[3].append(caplist[i][x])
                tickers[3].append(tlist[i][x])
            elif Rlist[i][x]['date'][0] < datetime(2021,6,30) and Rlist[i][x]['date'][0] > datetime(2020,6,30):
                ratios[4].append(Rlist[i][x])
                prices[4].append(Plist[i][x])
                returns[4].append(HPRlist[i][x])
                caps[4].append(caplist[i][x])
                tickers[4].append(tlist[i][x])
    return ratios,prices,returns,caps,tickers

sorted_lists = sortDate(c_ratio,c_prices,hprs,caps2,tickers)

c_ratio = sorted_lists[0]
c_prices = sorted_lists[1]
hprs = sorted_lists[2]
caps = sorted_lists[3]
tickers = sorted_lists[4]

for x in c_ratio:
    print(len(x))
for x in caps:
    print(len(x))

try:
    for x in range(len(hprs)):
        for i in range(len(hprs[x])):
            while max(hprs[x][i].values) > .5:
                try:
                    if hprs[x][i] > 50:
                        c_prices[x].pop(i)
                        c_ratio[x].pop(i)
                        hprs[x].pop(i)
                        caps[x].pop(i)
                        tickers[x].pop(i)
                except:
                    pass  
except:
    pass

for x in range(len(hprs)):
    for i in range(len(hprs[x])):
        if hprs[x][i] > 50:
            print(x,i)

T = [sum(x) for x in caps]
testCap = [x.copy() for x in caps]
for x in range(len(testCap)):
    for i in range(len(testCap[x])):
        testCap[x][i] = testCap[x][i]/T[x]

try:
    for x in range(len(caps)):
        for i in range(len(caps[x])):
            while max(testCap[x][i].values) > .5:
                try:
                    if testCap[x][i] > .5:
                        c_prices[x].pop(i)
                        c_ratio[x].pop(i)
                        hprs[x].pop(i)
                        caps[x].pop(i)
                        testCap[x].pop(i)
                        tickers[x].pop(i)
                except:
                    pass
except:
    pass

for x in range(len(hprs)):
    for i in range(len(hprs[x])):
        if testCap[x][i] > .5:
            print(x,i)

#put all the ratios together
c_ratios = [pd.concat(x) for x in c_ratio]

for x in c_ratios:
    x.reset_index(drop = True, inplace = True)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
le = LabelEncoder()
ohe = OneHotEncoder(sparse = False)

for x in c_ratios:
    x['sector'] = le.fit_transform(x['sector'])

ohedata = [(ohe.fit_transform(x['sector'].values.reshape( -1, 1 ) )).astype( int ) for x in c_ratios]

for i in range(len(ohedata)):
    for col in range( 0, ohedata[i].shape[1] - 1 ):
        c_ratios[i].insert( loc = col, column = 'sector_' + str( col + 1 ), value = ohedata[i][ :, col ])

for x in c_ratios:
    x.drop(['sector'],axis = 1,inplace = True)



#This code will sort and pick out the chosen stocks

#make the HPR into yearly dataframes
HPR = [pd.DataFrame(i) for i in hprs]

#chnge column names
for x in HPR:
    x.columns = (["returns"])



### restart here

ratios = []
#marge the returns and ratios together



def mergeR(i):
    x = pd.concat([c_ratios[i],HPR[i]],axis = 1)
    ratios.append(x)
for i in range(len(c_ratios)):
    mergeR(i)

market = [.0635,.2113,-.0524,.3102,.2089,.1632]

for i in range(len(ratios)):
    #binary variable on beating the market
    ratios[i]['binary m'] = np.where(ratios[i]["returns"]>market[i],1,0)

    #binary variable on stock going up
    ratios[i]['binary up down'] = np.where(ratios[i]["returns"]>0,1,0)

#isolate all of the years together

binary_market = [x['binary m'] for x in ratios]
binary_updown = [x['binary up down'] for x in ratios]

#concatenate all of the years together

Ymark = pd.concat(binary_market)
Yud = pd.concat(binary_updown)

#reset index
Yud.reset_index(drop = True,inplace = True)

Ymark.reset_index(drop = True,inplace =  True)

#drop them from tdataframes
#for x in ratios:
 #   x.drop(['binary m','binary up down','returns'],axis = 1,inplace = True)

#get rid of the useless columns
xnew =  [x.drop(['symbol','date', 'period'],axis = 1) for x in ratios]

#fillanas with 0
wdata = [x.fillna(0) for x in xnew]

from scipy.stats.mstats import winsorize


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
binaryVar = ['sector_1', 'sector_2', 'sector_3', 'sector_4', 'sector_5', 'sector_6',
       'sector_7', 'sector_8', 'sector_9', 'sector_10', 'sector_11','sector_12']

binary = [x[binaryVar] for x in wdata]

#up_majority = up[up['binary up down']==1]
#up_minority =up[up['binary up down']==0]

#scale the data
for x in wdata:
    x.drop(binaryVar,axis = 1,inplace = True)
    x.drop(['returns', 'binary m','binary up down'],axis = 1, inplace = True)

columns = wdata[0].columns

for x in wdata:
    for i in range(11,(len(columns) - 3)):
        x[columns[i]] = winsorize(x[columns[i]],limits = [.05,.05])

columns = wdata[0].columns
scaled = [scale.fit_transform(g) for g in wdata]

#needs to be transposed again to eventually turn into dataframe
inter = [x.T for x in scaled]
#turn into list to remove dataframe
scales = [x.tolist() for x in inter]

dfs = []
#this will make new dictionaries
for i in range(len(scaled)):
    x = dict(zip(columns,scales[i] ))
    dfs.append(x)

X = [pd.DataFrame(x) for x in dfs]

combined = []
for i in range(len(X)):
    x = pd.concat([X[i],binary[i]],axis = 1)
    combined.append(x)

for x in caps:
    print(len(x))
for x in combined:
    print(len(x))

x = pd.concat(combined)
x.reset_index(drop = True,inplace = True)

Yud.reset_index(drop = True,inplace = True)

#This code will test machine learning techniques on portfolio management, namely predicting stocks that beat the market based off of ratios


#dataframe with all x variables and beating the market y variable
market = pd.concat([Ymark,x],axis =1)
up = pd.concat([Yud,x],axis = 1)
#ud = pd.concat([Yud,X],axis = 1)

#differentiate between majority and minority y variaables
up_majority = up[up['binary up down']==1]
up_minority =up[up['binary up down']==0]
#code to resample
up_upsampled = resample(up_minority,replace = True,n_samples = len(up_majority),random_state = 123)
up_upsampled.to_csv("C:\\Users\\Lucas\\Documents\\market_dataU.csv")
#yet another repeat
market_majority = market[market['binary m']==0]
market_minority =market[market['binary m']==1]
market_upsampled = resample(market_minority,replace = True,n_samples = len(market_majority),random_state = 123)
market_upsampled.to_csv("C:\\Users\\Lucas\\Documents\\market_dataM.csv")
#bring the resampled minority and majority back together
M1 = pd.concat([up_majority,up_upsampled])
M2 = pd.concat([market_majority,market_upsampled])

#get the Y variables for both
Y1 = M1['binary up down'].values
Y2 = M2['binary m'].values

#drop the Y variables for both
M1.drop(['binary up down'],axis = 1,inplace = True)
M2.drop(['binary m'],axis =1, inplace = True)



#run pca
modelPCA = PCA(n_components = 6)
x = modelPCA.fit_transform(M1)
xM = modelPCA.fit_transform(M2 )
#change X variables into numpy arrays
x1 = x
X1 = x1.astype(np.float16)
x2 = xM
X2 = x2.astype(np.float32)

#split into training data and test data
X_train1, X_test1,y_train1,y_test1 =  model_selection.train_test_split( X1, Y1.ravel(), test_size = 0.25, random_state = 7 )
X_train2, X_test2,y_train2,y_test2 =  model_selection.train_test_split( X2, Y2.ravel(), test_size = 0.5, random_state = 7 )

from xgboost import XGBClassifier

modelU = XGBClassifier(gamma = 0, max_depth = 6,min_child_weight = 5)
modelU.fit(X_train1,y_train1)
predictions = modelU.predict(X_test1)
predictions2 = modelU.predict(X_train1)
testAccuracy = (accuracy_score(y_test1, predictions))
trainAccuracy = accuracy_score(y_train1,predictions2)
print(testAccuracy)
print(trainAccuracy)
print( confusion_matrix( y_test1, predictions ) )

from xgboost import XGBClassifier

modelM = XGBClassifier(gamma = 0, max_depth = 6,min_child_weight = 5)
modelM.fit(X_train2,y_train2)
predictions = modelM.predict(X_test2)
predictions2 = modelM.predict(X_train2)
testAccuracy = (accuracy_score(y_test2, predictions))
trainAccuracy = accuracy_score(y_train2,predictions2)
print(testAccuracy)
print(trainAccuracy)
print( confusion_matrix( y_test2, predictions ) )

kfold = 8

results = model_selection.cross_val_score( modelU, X1, Y1, cv = kfold )

print(results.mean())

### Backtest Portion

modelPCA.fit(M2)

PCAbyYear = [modelPCA.transform(x) for x in combined]

predictions = [modelM.predict(x) for x in PCAbyYear]

prob = [modelU.predict_proba(x) for x in PCAbyYear]
predictions = [x.reshape(len(x),1) for x in predictions]



for x in predictions:
    print(len(x))
for x in caps:
    print(len(x))

totalCap = [sum(x) for x in caps]
for x in range(len(predictions)):
    for i in range(len(predictions[x])):
        caps[x][i] = caps[x][i]/totalCap[x]

caps = [np.array(x) for x in caps]

from scipy.optimize import linprog

def noShortLinearConstructor(alphaScore, benchmarkW, securityOffset):
    maxWeights =benchmarkW + securityOffset
    minWeights = benchmarkW - securityOffset
    minWeights = np.where(minWeights < 0,0,minWeights)
    alphaScore = alphaScore * -1
    maxWeights = maxWeights.reshape(len(maxWeights),1)
    minWeights = minWeights.reshape(len(minWeights),1)
    alphaScore = alphaScore.reshape(1,len(alphaScore))
    bounds = np.array(list(zip(minWeights,maxWeights)))
    bounds = bounds.reshape(len(maxWeights),2)
    Aeq =  np.ones(len(alphaScore[0]))
    Aeq = Aeq.reshape(1,len(Aeq))
    Beq = np.array(1)
    Beq = Beq.reshape(1,1)

    weights = linprog(alphaScore,A_eq = Aeq,b_eq = Beq,bounds = bounds).x
    return weights

linearWeights = []

for i in range(len(predictions)):
    print("calculating weights")
    weights = noShortLinearConstructor(predictions[i],caps[i],.1)
    print("weights calculated")
    linearWeights.append(weights)
    print("weights collected")

linearWeights = [x.reshape(1,len(x)) for x in linearWeights]

HPR = [x.values for x in HPR]

caps = [x.reshape(len(x[0]),1) for x in caps]

for x in caps:
    print(x.shape)

for x in HPR:
    print(x.shape)

#this would be the returns assuming the financial years started at the same time

for i in range(len(HPR)):
    print("market")
    print( round((np.array(HPR[i]).T @ caps[i])[0][0],4))
    print("model")
    print(round((linearWeights[i] @ HPR[i])[0][0],4))

#analytics
    #get indicies returns
    #change it to month by month to be more accurate
    #and get more datapomits
#what risks drove returns

#residual alphas
#attribution


indexTick = ['IWD','IWF','IWN','IWO']

indicies = ['1000 value','1000 growth','2000 value','2000 growth']

indexTick[1]

indexP = []

for i in range(len(indexTick)):
    x = fmp.get_historical_price(indexTick[i])
    indexP.append(x)


indexP = [x[::-1] for x in indexP]

for x in range(len(indexP)):
    indexP[x].reset_index(drop = True, inplace = True)
    indexP[x].to_csv(("c://Users//Lucas//Documents//"+indexP[i]+".csv"))
### where to look at code

from datetime import datetime
from datetime import timedelta

#x = testDate[0][45]
#$x2 = testDate[0][0]
#max(testDate[4]) - x2

start = x2 + timedelta(days = 1)

rebalance = []
for i in range(60):
    x = start + timedelta(days = (27 + 27*i))
    rebalance.append(x)

for x in new_month_list:
    try:
        x.drop(['level_0','index'],axis = 1,inplace = True)
    except:
        pass

import math


#ratio2
#new_month_list

import math

#ratios
testR = [x.copy() for x in ratio2]

#prices
testP = [x.copy() for x in new_month_list]

for x in testP:
    x.reset_index(inplace = True)

dP = testP[0]['date'].values

testP[0]['date'][0]

dR = testR[0]['date'].values


#IWD 1000 value
#IWF 1000 growth
#IWN 2000 value
#IWO 2000 growth
