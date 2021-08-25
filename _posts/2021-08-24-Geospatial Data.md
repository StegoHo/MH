Geospatial data Analytics could come in handy when dealing with B2C corporations. Lots of companies possess geodata such as zip codes, addresses, or even GPS data from electronic devices. Hence, utilizing the data could be the key to understanding the pros and cons for the business.

 

Several kinds of analysis could be performed to assist when making decision. However, beforehand, there is some data transformation that needs to be done. Addresses should translate into latitude and longitude for projections on the map. Among all the method, I would recommend using Google API geocoding which converts addresses to geographic coordinates costing 5 dollars per thousand requests. There’s also chances that it could be free if the monthly charges are below 200. 



The will look like this:

```python
GEOCODE_BASE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

def geocode(address):
    params = urllib.parse.urlencode({"address":address, "key":api})
    url = GEOCODE_BASE_URL + '?' + params
	result = json.load(urllib.request.urlopen(url))
	if result["status"] in ["OK", "ZERO_RESULTS"]:
    	return result["results"]
	raise Exception(result["error_message"])
```

  

While finishing the transformation, some distance data could be obtained. On the other hand, packages such as folium could be very useful when visualizing it. Here is one easy example for some of the stores of MamaFisch in Taiwan. Users can scroll and drag intuitively and information can be looked up in hover and pop up.


<video width="600" height="400" controls>
  <source src="assets/images/MamaFisch.mp4" type="video/mp4">
</video>
