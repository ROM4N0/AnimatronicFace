#include <Servo.h>

Servo myservo1,myservo2,myservo3,myservo4,myservo5,myservo6,myservo7;

int pos = 90;
int x,y,m;
String s = "",input="";
void setup(){

  Serial.begin(9600);
  Serial.setTimeout(10);

  myservo1.attach(2);   //RED      UPPER LEFT
  myservo2.attach(3);   //ORANGE   UPPER RIGHT
  myservo3.attach(5);   //BLUE     LOWER LEFT
  myservo4.attach(6);   //PURPLE   LOWER RIGHT
  myservo5.attach(9);   //YELLOW   EYE HORIZONTAL
  myservo6.attach(11);  //BROWN    EYE VERTICAL
  myservo7.attach(12);  //GREEN    MOUTH

  myservo1.write(pos);
  myservo2.write(pos);
  delay(50);
  myservo3.write(pos);
  myservo4.write(pos);
  delay(50);
  myservo5.write(pos);
  myservo6.write(pos);

  myservo7.write(pos);


}

void loop(){
  while(Serial.available()){
    char c = Serial.read();
    if (c=='<'){
      input="";
    }else if (c=='>'){
      // input == "XxxxYyyyMmmm"
      if(input[0]=='X' && input[4]=='Y' && input[8]=='M'){
      x = input.substring(1,4).toInt();
      y = input.substring(5,8).toInt();
      m = input.substring(9).toInt();
      moveEyes(x,y);
      moveMouth(m);
      Serial.print("ACK <X");
      Serial.print(x);
      Serial.print("Y");
      Serial.print(y);
      Serial.print("M");
      Serial.print(m);
      Serial.println(">");
      input="";
      }else{
        Serial.print("Communication Error. Try again!");
      }
    }
    else{
      input+=c;
    }
  }
}

void moveEyes(int x,int y){

  if(x > 135 || y > 135 ){
    myservo5.write(pos);
    myservo6.write(pos);
    delay(50);
    myservo1.write(45);
    myservo2.write(135);
    delay(50);
    myservo3.write(135);
    myservo4.write(45);
  }else{

    myservo5.write(x); // x
    myservo6.write(y); // y
    delay(50);

    if (y > 90){
      myservo1.write(y);
      myservo2.write(180-y);
      delay(50);
      myservo3.write(pos);
      myservo4.write(pos);
    }else if (y < 90){
      myservo1.write(pos);
      myservo2.write(pos);
      delay(50);
      myservo3.write(y);
      myservo4.write(180-y);
    }else{
      myservo1.write(pos);
      myservo2.write(pos);
      delay(50);
      myservo3.write(pos);
      myservo4.write(pos);
    }
  }
}

void moveMouth(int m){
  if (-1 == m){
    myservo7.detach();
  } else if(-2==m){
    myservo7.attach(12);
  }else{
    myservo7.write(m);
  }
}

