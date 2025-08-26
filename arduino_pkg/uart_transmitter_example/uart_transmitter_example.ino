uint16_t sensor_data;
byte buf[2];

void setup(){
 Serial.begin(38400);
}

void loop(){
 sensor_data = random(0, 1024);

 buf[0] = (sensor_data >> 8) & 0xFF; 
 buf[1] = sensor_data & 0xFF;

 Serial.write(buf, 2);
 delay(100);
}
