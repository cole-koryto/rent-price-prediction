"""
This class contains the functions to extract all current rental listings from Trulia nationally.
It also has the functions to clean each attribute and add all other needed features.

By Cole Koryto
"""
import pprint
import pandas as pd
from bs4 import BeautifulSoup
import time
import censusgeocode as cg

class DataExtract:

    totalRentalDf = pd.DataFrame()

    # extracts all data for given fips code
    @staticmethod
    def extractData(driver, fipsCode):

        # loads the contents of the page into BeautifulSoup
        print(fipsCode)
        time.sleep(10)
        targetURL = f"https://www.trulia.com/for_rent/{fipsCode}_c/8_zm/"
        driver.get(targetURL)
        firstPageContent = driver.page_source
        soup = BeautifulSoup(firstPageContent, features="html.parser")

        # checks that program is not waiting for human response
        humanResponseTags = soup.findAll('div', attrs={"aria-label": "Human challenge"})
        if len(humanResponseTags) > 0:
            input("\n***Human input needed to continue***\nPress enter when ready")
            driver.get(targetURL)
            firstPageContent = driver.page_source
            soup = BeautifulSoup(firstPageContent, features="html.parser")

        # finds the number of pages of data on trulia.com (gets last page number at bottom of first page)
        try:
            totalPages = int(soup.findAll(attrs={'data-testid': 'pagination-page-link'})[-1].text)
        except Exception as e:
            print(f"No results for FIPS code: {fipsCode}")
            return

        print(f"Total pages {totalPages}")

        # extracts rental information from all available pages
        prices = []
        beds = []
        baths = []
        sqft = []
        addresses = []
        CONST_FETCHING_LOOP = 1
        print("Fetching Trulia rental data")

        # loops CONST_FETCHING_LOOP times to ensure all listings are retrieved, does this to make sure all listings are grabbed
        # this has to be done because Trulia randomly excludes some listings when you look up, so it much be run several times
        # to get all listings
        pagesProcessed = 0
        for i in range(0, CONST_FETCHING_LOOP):

            # loops over all pages for city
            for currentPageNum in range(1, totalPages + 1):
                # prints current page progress
                print("Loading... {0:.5} %".format(str(pagesProcessed / (totalPages * CONST_FETCHING_LOOP) * 100)))

                # sets up web driver for current page
                driver.get(targetURL + str(currentPageNum) + "_p/")
                currentPageContent = driver.page_source
                soup = BeautifulSoup(currentPageContent, features="html.parser")

                # navigates to each listing on a page
                for listing in soup.findAll(attrs={
                    'class': 'Grid__CellBox-sc-144isrp-0 SearchResultsList__WideCell-sc-14hv67h-2 kyKpOC fXGrSx'}):

                    # checks if this listing block is empty of data by checking if there is a data-testid attribute
                    if not listing.has_attr('data-testid'):
                        continue

                    # gets addresses by getting the correct data-testid attribute
                    if listing.find(attrs={'data-testid': 'property-address'}) != None:
                        tag = listing.find(attrs={'data-testid': 'property-address'})
                        attribute = tag['title']

                        # checks if address has already been put in list in another iteration
                        if attribute in addresses:
                            continue
                        else:
                            addresses.append(attribute)
                    else:
                        addresses.append("No Data")

                    # gets prices by getting the correct data-testid attribute
                    if listing.find(attrs={'data-testid': 'property-price'}) != None:
                        tag = listing.find(attrs={'data-testid': 'property-price'})
                        attribute = tag['title']
                        prices.append(attribute)
                    else:
                        prices.append("No Data")

                    # gets the number of beds by getting the correct data-testid attribute
                    if listing.find(attrs={'data-testid': 'property-beds'}) != None:
                        tag = listing.find(attrs={'data-testid': 'property-beds'})
                        attribute = tag['title']
                        beds.append(attribute)
                    else:
                        beds.append("No Data")

                    # gets the number of baths by getting the correct data-testid attribute
                    if listing.find(attrs={'data-testid': 'property-baths'}) != None:
                        tag = listing.find(attrs={'data-testid': 'property-baths'})
                        attribute = tag['title']
                        baths.append(attribute)
                    else:
                        baths.append("No Data")

                    # gets the square footage by getting the correct data-testid attribute
                    if listing.find(attrs={'data-testid': 'property-floorSpace'}) != None:
                        tag = listing.find(attrs={'data-testid': 'property-floorSpace'})
                        attribute = tag['title']
                        sqft.append(attribute)
                    else:
                        sqft.append("No Data")

                # increments pages processed
                pagesProcessed += 1

        # sets up data output with pandas
        column1 = pd.Series(addresses, name='Addresses')
        column2 = pd.Series(prices, name='Prices')
        column3 = pd.Series(beds, name='Beds')
        column4 = pd.Series(baths, name='Baths')
        column5 = pd.Series(sqft, name='Square Footage')
        df = pd.DataFrame(
            {'Addresses': column1, 'Prices': column2, 'Beds': column3, 'Baths': column4, 'Square Footage': column5})

        # appends county data to overall dataframe
        DataExtract.totalRentalDf = pd.concat([DataExtract.totalRentalDf, df], axis=0)

    # cleans dataset by removing unneeded instances
    @staticmethod
    def cleanData():

        print("\nCleaning data")

        # gets original dataset dataframe
        originalDf = pd.read_csv("Total Rental Results.csv")

        # removes instances in columns that have 'No Data'
        originalDf = originalDf[originalDf['Addresses'] != 'No Data']
        originalDf = originalDf[originalDf['Prices'] != 'No Data']
        originalDf = originalDf[originalDf['Beds'] != 'No Data']
        originalDf = originalDf[originalDf['Baths'] != 'No Data']
        originalDf = originalDf[originalDf['Square Footage'] != 'No Data']

        # removes instances with one or more blank values
        originalDf.dropna(inplace=True)

        # outputs new clean dataframe
        originalDf.to_csv("Clean Total Rental Results.csv", index=False)

    # takes overall rental dataframe and adds longitude and latitude
    @staticmethod
    def addLongAndLat():

        print("\nAdding longitude and latitude")
        totalDf = pd.read_csv("Clean Total Rental Results.csv")
        geoCodeResultList = []
        for numAddress, address in enumerate(totalDf.loc[:, 'Addresses']):
            print(f"Getting address {numAddress+1} of {len(totalDf.loc[:, 'Addresses'])}")
            try:
                result = cg.onelineaddress(address)
                if len(result) > 0:
                    geoCodeResultList.append([result[0]['coordinates']['x'], result[0]['coordinates']['y']])
                else:
                    geoCodeResultList.append([None, None])
            except Exception as e:
                print(f"Error caused by address {address}\n{e}")
                geoCodeResultList.append([None, None])

        # makes dataframe from results
        geoCodeResultsDf = pd.DataFrame(geoCodeResultList, columns=["Longitude", "Latitude"])

        # concatenates geocoder results to overall dataframe
        totalDf = pd.concat([totalDf, geoCodeResultsDf], axis=1)

        # outputs new file
        totalDf.to_csv("Clean Dataset with lat and long.csv", index=False)

    # drops all instances with missing values in any attribute
    @staticmethod
    def dropMissingAttributes():

        print("\nDropping instances missing attributes")

        # gets original dataset dataframe
        cleanWithAllAttsDf = pd.read_csv("Clean Dataset with lat and long.csv")

        # removes instances with one or more blank values
        cleanWithAllAttsDf.dropna(inplace=True)

        # remove instances that have the same address
        cleanWithAllAttsDf.drop_duplicates(subset=['Addresses'], keep='first', inplace=True)

        # outputs new clean dataframe
        cleanWithAllAttsDf.to_csv("Clean Dataset with latlog and no na.csv", index=False)


    # extracts numbers only from applicable columns
    @staticmethod
    def extractNumbersOnly():

        print("\nExtracting numbers")

        # reads in clean data with no missing attributes
        cleanWithAllAttsDf = pd.read_csv("Clean Dataset with latlog and no na.csv")

        # extracts numbers from columns
        cleanWithAllAttsDf['Beds_num'] = cleanWithAllAttsDf['Beds'].str.extract(r'([-+]?\d*\.\d+|[-+]?\d+)').fillna(1).astype(float)
        #cleanWithAllAttsDf['Beds_num'] = cleanWithAllAttsDf['Beds_num'].str.replace('', '1', regex=False)
        cleanWithAllAttsDf['Baths_num'] = cleanWithAllAttsDf['Baths'].str.extract(r'([-+]?\d*\.\d+|[-+]?\d+)').astype(float)
        cleanWithAllAttsDf['Square Footage_num'] = cleanWithAllAttsDf['Square Footage'].str.replace(r'[^\(\)0-9]*', '', regex=True)
        cleanWithAllAttsDf['Square Footage_num'] = cleanWithAllAttsDf['Square Footage_num'].str.replace(r'\(.*\)', '', regex=True).astype(float)
        cleanWithAllAttsDf['Prices_num'] = cleanWithAllAttsDf['Prices'].str.replace(r'[^\(\)0-9]*', '', regex=True).astype(float)

        # drops highest 1% of price
        cleanWithAllAttsDf = cleanWithAllAttsDf.loc[(cleanWithAllAttsDf['Prices_num'] < 10000), :]
        # drops highest 1% of sq ft
        cleanWithAllAttsDf = cleanWithAllAttsDf.loc[(cleanWithAllAttsDf['Square Footage_num'] < 10000), :]

        # outputs new clean dataframe
        cleanWithAllAttsDf.to_csv("FINAL Rental Results.csv", index=False)
