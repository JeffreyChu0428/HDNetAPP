#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

BLECharacteristic *pCharacteristic;
bool deviceConnected = false;

// 自訂 Service 和 Characteristic UUID（要與 JS 相同）
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

class MyServerCallbacks: public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
    Serial.println("Client connected");
  };

  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
    Serial.println("Client disconnected");
  }
};

void setup() {
  Serial.begin(115200);

  pinMode(41, INPUT); // Setup for leads off detection LO +
  pinMode(40, INPUT); // Setup for leads off detection LO -

  BLEDevice::init("ESP32_ECG_BLE"); // 藍牙裝置名稱
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ   |
                      BLECharacteristic::PROPERTY_WRITE  |
                      BLECharacteristic::PROPERTY_NOTIFY
                    );

  pCharacteristic->addDescriptor(new BLE2902());
  pService->start();
  pServer->getAdvertising()->start();
  Serial.println("BLE device is ready to connect");
}

void loop() {
  if (deviceConnected) {
    if((digitalRead(40) == 1)||(digitalRead(41) == 1)){
      String ecgStr = String(0);
      pCharacteristic->setValue(ecgStr.c_str());
      pCharacteristic->notify();
      delay(4);
    }
    else{
      float voltage = analogRead(A0) * 3.3 / 4095.0;
      String ecgStr = String(voltage, 3);
      pCharacteristic->setValue(ecgStr.c_str());
      pCharacteristic->notify();
      delay(4); // 250 Hz 頻率發送
    }
  }
}
