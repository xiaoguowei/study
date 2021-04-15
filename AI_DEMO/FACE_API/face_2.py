import http.client, urllib, base64, json

###############################################
#### Update or verify the following values. ###
###############################################

# Replace the subscription_key string value with your valid subscription key.
subscription_key = '12bd983ea40c4f968a72db5e45f0c42b'

# Replace or verify the region.
#
# You must use the same region in your REST API call as you used to obtain your subscription keys.
# For example, if you obtained your subscription keys from the westus region, replace 
# "westcentralus" in the URI below with "westus".
#
# NOTE: Free trial subscription keys are generated in the westcentralus region, so if you are using
# a free trial subscription key, you should not need to change this region.
uri_base = 'westus.api.cognitive.microsoft.com'

# Request headers.
headers = {
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': subscription_key,
}

# Request parameters.
params = urllib.parse.urlencode({
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'true',
    'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
})

# The URL of a JPEG image to analyze.
body = "{'url':'https://cdn2.ettoday.net/images/4143/d4143526.jpg'}"

#https://westus.api.cognitive.microsoft.com/face/v1.0

try:
    # Execute the REST API call and get the response.
    conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
    conn.request("POST", "/face/v1.0/detect?%s" % params, body, headers)
    response = conn.getresponse()
    data = response.read()

    # 'data' contains the JSON data. The following formats the JSON data for display.
    parsed = json.loads(data)
    print ("Response:")
    print (json.dumps(parsed, sort_keys=True, indent=2))
    print (parsed[0]["faceAttributes"]["smile"])
    print (parsed[0]["faceAttributes"]["gender"])
    print (parsed[0]["faceAttributes"]["age"])
    conn.close()
    #start here     
    x = parsed[0]["faceAttributes"]["smile"]
    if x > 0.6:
        print ("welcome")
    else:
        print ("smile not enough!")

except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))

