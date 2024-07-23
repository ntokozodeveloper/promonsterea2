import React, { useState, useEffect } from 'react';
import { getHistoricalData, executeTrade } from './services/api';
import io from 'socket.io-client';

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
//import Home from './pages/Home';
import About from './pages/About';
import NotFound from './pages/NotFound';

// Add line to route homepage : <Route exact path="/" component={Home} />

function App() {
  return (
    <Router>
      <Switch>
        <Route path="/about" component={About} />
        <Route component={NotFound} />
      </Switch>
    </Router>
  );
}


//Deriv Connection
const WebSocket = require('ws');
const DerivAPI = require('@deriv/deriv-api/dist/DerivAPI');

const app_id = 61999; // Replace with your app_id or leave as 1089 for testing.
const websocket = new WebSocket(`wss://ws.derivws.com/websockets/v3?app_id=${app_id}`);
const api = new DerivAPI({ connection: websocket });
const basic = api.basic;
const ping_interval = 12000; // it's in milliseconds, which equals to 120 seconds
let interval;


basic.ping().then(console.log);

const newLocal = new WebSocket('wss://ws.derivws.com/websockets/v3?app_id=61999');

//Pass

// subscribe to `open` event
websocket.addEventListener('open', (event) => {
  console.log('websocket connection established: ', event);
  const sendMessage = JSON.stringify({ ping: 1 });
  websocket.send(sendMessage);

  // to Keep the connection alive
  interval = setInterval(() => {
    const sendMessage = JSON.stringify({ ping: 1 });
    websocket.send(sendMessage);
  }, ping_interval);
});

// subscribe to `message` event
websocket.addEventListener('message', (event) => {
  const receivedMessage = JSON.parse(event.data);
  console.log('new message received from server: ', receivedMessage);
});

// subscribe to `close` event
websocket.addEventListener('close', (event) => {
  console.log('websocket connectioned closed: ', event);
  clearInterval(interval);
});

// subscribe to `error` event
websocket.addEventListener('error', (event) => {
  console.log('an error happend in our websocket connection', event);
});

//**Socket, Real time updates */

const socket = io('http://localhost:3000');




export default App;

