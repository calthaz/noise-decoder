let startPrediction = null;
document.addEventListener("DOMContentLoaded", function(event) {
    const sw = document.getElementById("sound-wrapper");
    
    //////////////////////change to https for deployment////////////////////////////
    socket = io.connect('http://localhost:5001/web', {
        reconnection: false,
        transports: ['websocket']
    });
    
    socket.on('connect', () => {
        alert("connected")
        console.log('Connected');
    });
    
    socket.on('disconnect', () => {
        alert("disconnected")
        console.log('Disconnected');
    });
    
    socket.on('connect_error', (error) => {
        alert("connect error "+error)
        console.log('Connect error! ' + error);
    });
    
    socket.on('connect_timeout', (error) => {
        alert("time out")
        console.log('Connect timeout! ' + error);
    });
    
    socket.on('error', (error) => {
        alert(error);
        console.log('Error! ' + error);
    });
    
    // Update image and text data based on incoming data messages
    socket.on('words', (msg) => {
        //console.log("receive server2web");
        console.log(msg);
        sw.innerHTML += msg;
        window.scrollTo(0, sw.scrollHeight)
    });

    const btn = document.getElementById("start-prediction")
    startPrediction = ()=>{
        console.log("start prediction 60 seconds")
        //socket deconnects in 60 seconds
        socket.emit('predict', 60)
    }
    btn.addEventListener("click", startPrediction)
});

