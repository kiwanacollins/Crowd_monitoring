require('dotenv').config();

const express = require('express');
const path = require('path');
const { MongoClient } = require('mongodb');
const app = express();
const port = process.env.PORT || 3000;
const mongoose = require('mongoose');

const cors = require('cors');

const uri = process.env.MONGO_URI || process.env.mongo_URL || 'mongodb://localhost:27017';

const client = new MongoClient(uri);
let collection;

async function connectToDatabase() {
    try {
        await client.connect();
        console.log('MongoDB connected successfully');
        const db = client.db('home');
        collection = db.collection('blogs');
    } catch (err) {
        console.error('Failed to connect to MongoDB:', err);
        process.exit(1); 
    }
}
app.use(cors());
app.use(express.static('admin'));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Security headers
app.use((req, res, next) => {
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('X-XSS-Protection', '1; mode=block');
    next();
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        service: 'crowd-dashboard',
        version: process.env.npm_package_version || '2.0.0',
    });
});

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'admin', 'admin.html'));
});
app.use(express.static('public'));


app.get('/crowd', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});
app.get('/data', async (req, res) => {
    console.log('Received request to fetch data');
    try {
        console.log('Fetching data from MongoDB...');
        const data = await collection.find({}).sort({ timestamp: -1 }).limit(50).toArray();
        console.log('Data fetched successfully. Number of documents:', data.length);

        let maxCrowd = 0;
        let totalCrowd = 0;
        let preferredShop = { lat: '', lng: '', count: Infinity };
        
        data.forEach(item => {
            if (item.count > maxCrowd) {
                maxCrowd = item.count;
            }
            totalCrowd += item.count;
            if (item.count < preferredShop.count) {
                preferredShop = { lat: item.coordinates.latitude, lng: item.coordinates.longitude, count: item.count };
            }
        });
        
        const averageCrowd = totalCrowd / data.length || 0;

        res.json({
            data,
            maxCrowd,
            averageCrowd: averageCrowd.toFixed(2),
            preferredShop
        });
    } catch (err) {
        console.error('Error fetching data from MongoDB:', err);
        res.status(500).json({ error: 'Error fetching data from MongoDB' });
    }
});

app.get('/history', async (req, res) => {
    const { lat, lng, minutes } = req.query;
    const timeRange = new Date(Date.now() - minutes * 60 * 1000); 

    try {
        console.log(`Fetching historical data for lat: ${lat}, lng: ${lng}, minutes: ${minutes}`);
        const data = await collection.find({
            "coordinates.latitude": parseFloat(lat),
            "coordinates.longitude": parseFloat(lng),
            timestamp: { $gte: timeRange }
        }).sort({ timestamp: -1 }).toArray();
        console.log('Historical data fetched successfully. Number of documents:', data.length);
        res.json(data);
    } catch (err) {
        console.error('Error fetching historical data from MongoDB:', err);
        res.status(500).json({ error: 'Error fetching historical data from MongoDB' });
    }
});
 
app.post('/update_data', async (req, res) => {
    const { coordinates, count } = req.body;

    try {
        console.log('Inserting new data into MongoDB:', { coordinates, count });
        const result = await collection.insertOne({
            coordinates,
            count,
            timestamp: new Date()
        });
        console.log("Data inserted into MongoDB:", result);

        const data = await collection.find({}).sort({ timestamp: -1 }).limit(50).toArray();
        let maxCrowd = 0;
        let totalCrowd = 0;
        let preferredShop = { lat: '', lng: '', count: Infinity };

        data.forEach(item => {
            if (item.count > maxCrowd) {
                maxCrowd = item.count;
            }
            totalCrowd += item.count;
            if (item.count < preferredShop.count) {
                preferredShop = { lat: item.coordinates.latitude, lng: item.coordinates.longitude, count: item.count };
            }
        });

        const averageCrowd = totalCrowd / data.length || 0;

        res.json({
            status: "success",
            maxCrowd,
            averageCrowd: averageCrowd.toFixed(2),
            preferredShop
        });
    } catch (err) {
        console.error("Error updating MongoDB:", err);
        res.status(500).json({ error: "Error updating MongoDB" });
    }
});
const HistorySchema = new mongoose.Schema({
    count: Number,
    coordinates: {
        latitude: Number,
        longitude: Number
    },
    timestamp: { type: Date }
});

const History = mongoose.model('History', HistorySchema);

app.get('/history', async (req, res) => {
    const { shop, time } = req.query;

    if (!shop || !time) {
        return res.status(400).json({ error: 'Shop and time parameters are required.' });
    }

    const currentTime = new Date();
    const pastTime = new Date(currentTime - time * 60 * 1000);  

    const shopCoordinates = {
        'Zudio': { latitude: 18.5204, longitude: 73.8567 },
        'Pheonix Mall': { latitude: 18.5250, longitude: 73.8567 },
        'Kaka Halwai': { latitude: 18.5369, longitude: 73.8567 },
        'Shaniwar Peth': { latitude: 18.5650, longitude: 73.8567 },
        'Starbucks': { latitude: 18.5850, longitude: 73.8567 }
        
    };

    const coordinates = shopCoordinates[shop];

    if (!coordinates) {
        return res.status(404).json({ error: 'Shop not found.' });
    }

    try {
        const history = await History.find({
            'coordinates.latitude': coordinates.latitude,
            'coordinates.longitude': coordinates.longitude,
            timestamp: { $gte: pastTime, $lte: currentTime }
        }).sort({ timestamp: -1 }); 

        if (history.length === 0) {
            return res.json({ message: 'No data available for the selected time range.' });
        }
        res.json(history);
    } catch (error) {
        console.error('Error fetching history:', error);
        res.status(500).json({ error: 'Server error' });
    }
});

mongoose.connect(process.env.MONGO_URI || process.env.mongo_URL || 'mongodb://localhost:27017', {
    useNewUrlParser: true,
    useUnifiedTopology: true
}).then(() => {
    console.log('Connected to MongoDB');
}).catch(err => {
    console.error('MongoDB connection failed:', err);
});

const userSchema = new mongoose.Schema({
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true }
});

const User = mongoose.model('User', userSchema);

const shopSchema = new mongoose.Schema({
    shopName: String,
    coordinates: String,
    videoFeed: String
});
const Shop = mongoose.model('Shop', shopSchema);

app.use(express.static(path.join(__dirname, 'admin')));

// app.get('/', (req, res) => {
//     res.sendFile(path.join(__dirname, 'admin', 'admin.html'));
// });

app.get('/addShop', (req, res) => {
    res.sendFile(path.join(__dirname, 'admin', 'shopDetails.html'));
});

app.post('/signup', async (req, res) => {
    const { email, password } = req.body;

    try {
        const newUser = new User({ email, password });
        await newUser.save();
        res.redirect('/crowd'); 
    } catch (error) {
        console.error('Error registering user:', error);
        res.status(500).send('Error registering user');
    }
});

app.post('/signin', async (req, res) => {
    const { email, password } = req.body;

    try {
        const user = await User.findOne({ email });
        if (!user) return res.status(404).send('User not found');

        if (password === user.password) {
            res.redirect('/crowd'); 
        } else {
            res.status(401).send('Invalid password');
        }
    } catch (error) {
        console.error('Error logging in:', error);
        res.status(500).send('Error logging in');
    }
});

app.post('/addShop', async (req, res) => {
    try {
        const newShop = new Shop({
            shopName: req.body.shopName,
            coordinates: req.body.coordinates,
            videoFeed: req.body.videoFeed
        });
        await newShop.save();
        res.status(201).send('Shop details saved successfully!');
    } catch (error) {
        console.error('Error saving shop details:', error);
        res.status(500).send('Error saving shop details');
    }
});
app.get('/shops', async (req, res) => {
    try {
        const shops = await Shop.find(); 
        res.status(200).json(shops); 
    } catch (error) {
        console.error('Error fetching shop details:', error);
        res.status(500).send('Error fetching shop details');
    }
});
const crowdSchema = new mongoose.Schema({
    count: Number,
    coordinates: {
        latitude: Number,
        longitude: Number
    },
    timestamp: String
});

const Crowd = mongoose.model('Crowd', crowdSchema);

app.get('/api/top-shops', async (req, res) => {
    try {
        const topShops = await Crowd.aggregate([
            { $group: { _id: "$coordinates", totalCrowd: { $sum: "$count" } } },
            { $sort: { totalCrowd: -1 } },
            { $limit: 5 }
        ]);
        res.json(topShops);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.get('/api/historical-data', async (req, res) => {
    const { latitude, longitude } = req.query;
    try {
        const historicalData = await Crowd.find({
            "coordinates.latitude": latitude,
            "coordinates.longitude": longitude
        }).sort({ timestamp: 1 }).limit(50); 
        res.json(historicalData);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.get('/api/heatmap', async (req, res) => {
    try {
        const heatmapData = await Crowd.find({}).select('coordinates count');
        res.json(heatmapData);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});
// ── Detection Session Storage (FYP evaluation data) ──────────────────────

const detectionSessionSchema = new mongoose.Schema({
    camera_id:     { type: String, required: true, index: true },
    label:         String,
    started_at:    { type: Date, default: Date.now },
    ended_at:      Date,
    count_in:      { type: Number, default: 0 },
    count_out:     { type: Number, default: 0 },
    peak_count:    { type: Number, default: 0 },
    avg_fps:       Number,
    ground_truth:  Number,   // manual ground-truth count for MAE evaluation
    notes:         String
});

const detectionEventSchema = new mongoose.Schema({
    camera_id:   { type: String, required: true, index: true },
    session_id:  { type: mongoose.Schema.Types.ObjectId, ref: 'DetectionSession', index: true },
    direction:   { type: String, enum: ['in', 'out'], required: true },
    track_id:    Number,
    timestamp:   { type: Date, default: Date.now }
});

const DetectionSession = mongoose.model('DetectionSession', detectionSessionSchema);
const DetectionEvent   = mongoose.model('DetectionEvent',   detectionEventSchema);

// POST /api/sessions — called by Python API when a camera session starts/ends
app.post('/api/sessions', async (req, res) => {
    try {
        const session = new DetectionSession(req.body);
        await session.save();
        res.status(201).json(session);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// GET /api/sessions — list all sessions (optionally filter by camera)
app.get('/api/sessions', async (req, res) => {
    try {
        const filter = req.query.camera_id ? { camera_id: req.query.camera_id } : {};
        const sessions = await DetectionSession.find(filter).sort({ started_at: -1 }).limit(100);
        res.json(sessions);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// GET /api/sessions/:id — single session detail
app.get('/api/sessions/:id', async (req, res) => {
    try {
        const session = await DetectionSession.findById(req.params.id);
        if (!session) return res.status(404).json({ error: 'Session not found' });
        res.json(session);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// PATCH /api/sessions/:id — update session (e.g., add ground_truth after manual count)
app.patch('/api/sessions/:id', async (req, res) => {
    try {
        const session = await DetectionSession.findByIdAndUpdate(
            req.params.id, req.body, { new: true, runValidators: true }
        );
        if (!session) return res.status(404).json({ error: 'Session not found' });
        res.json(session);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// GET /api/sessions/:id/events — get all crossing events for a session
app.get('/api/sessions/:id/events', async (req, res) => {
    try {
        const events = await DetectionEvent.find({ session_id: req.params.id }).sort({ timestamp: 1 });
        res.json(events);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// POST /api/events — log a single crossing event
app.post('/api/events', async (req, res) => {
    try {
        const event = new DetectionEvent(req.body);
        await event.save();
        res.status(201).json(event);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// GET /api/evaluation — MAE report: sessions with ground_truth set
app.get('/api/evaluation', async (req, res) => {
    try {
        const sessions = await DetectionSession.find({ ground_truth: { $exists: true, $ne: null } });
        const results = sessions.map(s => {
            const predicted = s.count_in;
            const gt        = s.ground_truth;
            return {
                session_id: s._id,
                camera_id:  s.camera_id,
                started_at: s.started_at,
                count_in:   predicted,
                ground_truth: gt,
                abs_error:  Math.abs(predicted - gt),
                pct_error:  gt > 0 ? ((Math.abs(predicted - gt) / gt) * 100).toFixed(2) : null
            };
        });
        const mae = results.length
            ? (results.reduce((sum, r) => sum + r.abs_error, 0) / results.length).toFixed(2)
            : null;
        res.json({ mae, n: results.length, sessions: results });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

connectToDatabase().then(() => {
    app.listen(port, () => {
        console.log(`Server is running on http://localhost:${port}`);
    });
});

