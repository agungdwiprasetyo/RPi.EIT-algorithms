import requests

class API(object):
	def __init__(self, host, port):
		super(API, self).__init__()
		self.headers = {"Authorization": "coegsekali", "Content-Type": "application/json"}
		self.host = str(host)+":"+str(port)

	def postImage(self, filename, kerapatan, algoritma, dataVolt):
		data = '{"nama":"'+str(filename)+'","data": "'+str(dataVolt)+'","algoritma":"'+str(algoritma)+'","kerapatan":"'+str(kerapatan)+'"}'
		url = self.host+'/image'
		response = requests.post(url, data, headers=self.headers)
		print(response.text)

	def postData(self):
		url = self.host+'/data'
		