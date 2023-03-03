from SoccerNet.Downloader import SoccerNetDownloader
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

#mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="/nfs/data/soccernet")
#mySoccerNetDownloader.downloadDataTask(task="calibration", split=["train","valid","test"])

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="data/datasets")
mySoccerNetDownloader.downloadDataTask(task="calibration", split=["train","valid","test"])