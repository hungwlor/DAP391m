from heat_map import heat_map
from counting_people import counting_people
if __name__ == "__main__":
    while True:
        print('Please choose module.')
        print('Choose 1 for Heat map')
        print('Choose 2 for Counting People')
        print('Choose 0 for Exit')
        option=int(input('Your choice: '))
        if option==1:
            heat_map()
        elif option==2:
            counting_people()
        elif option==0:
            break