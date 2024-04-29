const express=require("express");
const app=express()
const http=require('http').createServer(app)
const Port=process.env.PORT|| 3005;
http.listen(Port,()=>{
    console.log(`Listening on Port ${Port}`)
});
app.use(express.static(__dirname+'/public'))
app.get("/",(req,res)=>{
   res.sendFile(__dirname+ "/index.html")
})