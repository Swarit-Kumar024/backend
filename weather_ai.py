import weather_service

latitude = 22.5726
longitude = 88.3639

print("WB: hi! its weather_gpt here what would you like to know today")
x = input("You: ")

# get_weather() returns TWO values: (selected_weather_data, decoded_question).
# Unpack both here -- assigning to a single variable would make `data` the
# whole tuple instead of just the weather dict, which is what caused the
# AttributeError: 'tuple' object has no attribute 'get'
data, decoded = weather_service.get_weather(latitude, longitude, x)

# No need to call question_decoder(x) again -- get_weather() already ran it
# internally and returned the result as `decoded` above.

print("WB:", weather_service.natural_response(data, x, decoded))
