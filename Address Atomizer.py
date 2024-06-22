import pandas as pd
from collections import defaultdict
import re

def group_cities_by_state(file_path):
    # read the csv file
    df = pd.read_csv(file_path)
    # create a dictionary to store the states and their cities
    states_cities = defaultdict(list)

    # loop through each row in the dataframe
    for i, row in df.iterrows():
        state = row['state_id']
        city = row['city']
        # add the city to the list of cities for the corresponding state
        states_cities[state].append(city)
    # convert the dictionary to a list of lists
    states_cities = [[state, cities] for state, cities in states_cities.items()]

    return states_cities

# Call the function
state_city = group_cities_by_state("./uscities.csv")

def atomize_address(file_path, cities):
    # read the csv file and assign it to the variable df
    df = pd.read_csv(file_path)
    
    #for showing progress on lager data sets
    percentage = 0
    
    # loop through each row in the 'Addresses' column using the method iterrows()
    for i, row in df.iterrows():
        
        #indicates progress for larger data set. 
        if (round(i/df.shape[0]*100) != percentage) and (round(i/df.shape[0]*100) % 5 == 0):
            print(f"{percentage}% of rows processed")
        percentage = round(i/df.shape[0]*100) 
        
        address = row['Addresses']
        
        # get the last 5 characters from the address
        zip_code = address[-5:]
        
        # get the two letters state code from the address
        state = address[-8:-6]
        
        # update the 'Zip Code' column with the extracted zip code
        df.at[i, 'Zip Code'] = zip_code
        
        #update the 'State' column with the extracted state code
        df.at[i, 'State'] = state
        
        #initialize the city_match variable 
        city_match = None
        
        #loop through the state_city list to check if the address contain any of the cities
        for state_cities in cities:
            #check if the state match the current row state
            if state_cities[0] == state:
                for city in state_cities[1]:
                    if re.search(r"(\b" + city + r"\b)", address, re.IGNORECASE):
                        city_match = city
                        break
            if city_match:
                df.at[i, 'City'] = city_match
        if isinstance(city_match, str):      
            
            #remove the matched city name from the address
            address_no_city = re.sub(city_match, "", address)
            
            #update the "Street Address" column with the remaining address text
            df.at[i, "Street Address"] = address_no_city.split(' ,')[0]

    # drop the "Addresses" column
    df = df.drop("Addresses", axis=1)
    # print the dataframe info
    print(df.info())
    # save the modified dataframe to a new csv file
    df.to_csv("./output.csv", index=False)

# call the function
atomize_address("./Rental Results 1k.csv", state_city)


