/************************************************************
 *  Authors:      Ferrari, Lovo, Sabatti                    *
 *  Date:         2020/11/04                                *
 *  Macro:        Tjunction_Reader                          *
 *  Rev:          0                                         *
 *  Description:  this macro reads the the output voltages  *      
 *                of the two photodiode circuits in the     *
 *                T-Junction experiment and returns these   *
 *                two voltages and the time of the measures *
 ************************************************************/


void setup() {
  Serial.begin(9600);
}

void loop() {
  
  int pause = 10;
  
  // Data acquisition  
  int sensorValue0    = analogRead(A0);                         // reading (sensor 0)
  int sensorValue1    = analogRead(A1);                         // reading (sensor 1)
  long CurrentTime     = millis();                               // reading time [ms]
  
//   Data output
  Serial.print("A0");
  Serial.println(sensorValue0);
  Serial.print("A1");
  Serial.println(sensorValue1);
  Serial.print("TT");
  Serial.println(CurrentTime);
  
  delay(pause); // pause [ms] beetween two following measures

}
