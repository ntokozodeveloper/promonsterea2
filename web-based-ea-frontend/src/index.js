import React, { useEffect, useState } from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import axios from 'axios';
import styled from 'styled-components';


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

// Get "/Trade" Route and Style it 

const Container = styled.div`
    max-width: 600px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f4f4f4;
    border-radius: 8px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
`;

const Header = styled.h1`
    text-align: center;
    color: #333;
`;

const Item = styled.p`
    margin: 10px 0;
    padding: 10px;
    background-color: #fff;
    border-radius: 4px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const TradeResults = () => {
    const [tradeData, setTradeData] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get('http://localhost:3001/trade');
                setTradeData(response.data);
            } catch (error) {
                console.error('Error fetching trade data', error);
            }
        };

        fetchData();
    }, []);

    if (!tradeData) {
        return <div>Loading...</div>;
    }

    return (
        <Container>
            <Header>Trade Results</Header>
            <Item>Status: {tradeData.status}</Item>
            <Item>Message: {tradeData.message}</Item>
            <Item>Entry Price: {tradeData.entry_price}</Item>
            <Item>Stop Loss: {tradeData.stop_loss}</Item>
            <Item>Take Profit: {tradeData.take_profit}</Item>
            <Item>Probability: {tradeData.probability}</Item>
            <Item>Recommended Lot Sizes: {tradeData.recommended_lot_sizes.join(', ')}</Item>
            <Item>Timeframe Displayed: {tradeData.timeframe_displayed}</Item>
        </Container>
    );
};

export default TradeResults;



const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
