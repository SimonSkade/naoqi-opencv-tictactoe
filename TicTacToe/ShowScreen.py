import qi
import time
from naoqi import ALProxy
import webbrowser


class ShowScreen(object):

    def __init__(self, session, ip):
        self.__ip = ip
        try:
            self.__tablet = session.service("ALTabletService")
            self.__tablet.enableWifi()
            # tablet = session.service("ALTabletService")
            # tablet.enableWifi()
            # self.__tablet.configureWifi("wpa", "Thunder.kW", "Get.Out.Of:Hotspot")
            # self.__tablet.connectWifi("Thunder.kW")
            print(self.__tablet.getWifiStatus())
            #tablet.playVideo("https://youtu.be/M3pKKlSUPMk")
            #print tablet.getVideoPosition(), " / ", tablet.getVideoLength()
            #time.sleep(200)
            #tablet.stopVideo()
        except Exception as e:
            print(e)

    def show_screen(self, params):
        url = "http://" + "198.18.0.1" + "/apps/tic-tac-toe/index.html"
        url_flipped = "http://" + self.__ip + "/apps/tic-tac-toe/html/index-flipped.html"
        self.__tablet.showWebview(url + params)
        webbrowser.open(url_flipped+params, new=0, autoraise=True)

    def hide_screen(self):
        self.__tablet.hideWebview()
