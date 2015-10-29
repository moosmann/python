# Script to calculate air temperature from ELR

# First import required modules
import sys , os # Operating system modules

# Now define our function . Variables in brackets are the inputs

def model ( temperature , altitude ):

    #calculate modelled temperature
    newtemperature = temperature - 0.00649 * altitude
    # return this value to the script
    return newtemperature

# Now let's use our function to find some values
print model(283.15 , 1000)

print model(283.15 , 3500)

print model(271.15 , 712)

# When finished , exit the script
exit(0)
