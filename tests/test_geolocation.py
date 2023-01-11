from mats_util.geolocation import satellite as satellite
import datetime as DT

def test_satellite():
    date = DT.datetime(2022,11,24,13,30,0)#        2022-11-23 08:20:00                  
    satlat,satlon,satLT,nadir_sza,nadir_mza,TPlat,TPlon,TPLT,TPsza,TPssa = satellite.get_position(date)
    date = DT.datetime(2022,12,11,13,30,0)#        2022-11-23 08:20:00                  
    satlat,satlon,satLT,nadir_sza,nadir_mza,TPlat,TPlon,TPLT,TPsza,TPssa = satellite.get_position(date)
    date = DT.datetime(2022,12,11,13,30,0)#        2022-11-23 08:20:00                  
    satlat_2,satlon_2,satLT_2,nadir_sza_2,nadir_mza_2,TPlat_2,TPlon_2,TPLT_2,TPsza_2,TPssa_2 = satellite.get_position(date,database=True)
    print('test')

if __name__ == "__main__":
    test_satellite()