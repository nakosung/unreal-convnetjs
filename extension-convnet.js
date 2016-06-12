"use strict";

const num_agents = 10;
const map_scale = Math.sqrt(num_agents);
const sensor_range = 85 * 4;
    
const scale = 0.6 / map_scale
const canvas = {
    width: 700 * map_scale,
    height: 500 * map_scale
}

const chartcanvas = {
    width : 400,
    height : 150
}

let spec = `var num_inputs = 36; // 9 eyes, each sees 4 numbers (wall, green, red thing, other agent proximity)
var num_actions = 5; // 5 possible angles agent can turn
var temporal_window = 1; // amount of temporal memory. 0 = agent lives in-the-moment :)
var network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;

// the value function network computes a value of taking any of the possible actions
// given an input state. Here we specify one explicitly the hard way
// but user could also equivalently instead use opt.hidden_layer_sizes = [20,20]
// to just insert simple relu hidden layers.
var layer_defs = [];
layer_defs.push({ type: 'input', out_sx: 1, out_sy: 1, out_depth: network_size });
layer_defs.push({ type: 'fc', num_neurons: 50, activation: 'relu' });
layer_defs.push({ type: 'fc', num_neurons: 50, activation: 'relu' });
layer_defs.push({ type: 'regression', num_neurons: num_actions });

// options for the Temporal Difference learner that trains the above net
// by backpropping the temporal difference learning rule.
var tdtrainer_options = { learning_rate: 0.001, momentum: 0.0, batch_size: 64, l2_decay: 0.01 };

var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 30000 * num_agents;
opt.start_learn_threshold = 1000 * num_agents;
opt.gamma = 0.9;
opt.learning_steps_total = 200000;
opt.learning_steps_burnin = 3000;
opt.epsilon_min = 0.05;
opt.epsilon_test_time = 0.05;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;
opt.flip_interval = 100;

var brain = new deepqlearn.Brain(num_inputs, num_actions, opt); brain // woohoo`

const I = require('instantiator')

function main() {
    let editor;  
    let alive = true;
    global.convnetjs = require('./convnetjs/build/convnet')
    global.cnnutil = require('./convnetjs/build/util')
    global.cnnvis = require('./convnetjs/build/vis')
    global.deepqlearn = require('./convnetjs/build/deepqlearn')
    let _ = require('lodash')

    // A 2D vector utility
    var Vec = function (x, y) {
        this.x = x;
        this.y = y;
    }
    Vec.prototype = {

        // utilities
        dist_from: function (v) { return Math.sqrt(Math.pow(this.x - v.x, 2) + Math.pow(this.y - v.y, 2)); },
        length: function () { return Math.sqrt(Math.pow(this.x, 2) + Math.pow(this.y, 2)); },

        // new vector returning operations
        add: function (v) { return new Vec(this.x + v.x, this.y + v.y); },
        sub: function (v) { return new Vec(this.x - v.x, this.y - v.y); },
        rotate: function (a) {  // CLOCKWISE
            return new Vec(this.x * Math.cos(a) + this.y * Math.sin(a),
                -this.x * Math.sin(a) + this.y * Math.cos(a));
        },

        // in place operations
        scale: function (s) { this.x *= s; this.y *= s; },
        normalize: function () { var d = this.length(); this.scale(1.0 / d); }
    }

    // line intersection helper function: does line segment (p1,p2) intersect segment (p3,p4) ?
    var line_intersect = function (p1, p2, p3, p4) {
        var denom = (p4.y - p3.y) * (p2.x - p1.x) - (p4.x - p3.x) * (p2.y - p1.y);
        if (denom === 0.0) { return false; } // parallel lines
        var ua = ((p4.x - p3.x) * (p1.y - p3.y) - (p4.y - p3.y) * (p1.x - p3.x)) / denom;
        var ub = ((p2.x - p1.x) * (p1.y - p3.y) - (p2.y - p1.y) * (p1.x - p3.x)) / denom;
        if (ua > 0.0 && ua < 1.0 && ub > 0.0 && ub < 1.0) {
            var up = new Vec(p1.x + ua * (p2.x - p1.x), p1.y + ua * (p2.y - p1.y));
            return { ua: ua, ub: ub, up: up }; // up is intersection point
        }
        return false;
    }

    var line_point_intersect = function (p1, p2, p0, rad) {
        var v = new Vec(p2.y - p1.y, -(p2.x - p1.x)); // perpendicular vector
        var d = Math.abs((p2.x - p1.x) * (p1.y - p0.y) - (p1.x - p0.x) * (p2.y - p1.y));
        d = d / v.length();
        if (d > rad) { return false; }

        v.normalize();
        v.scale(d);
        var up = p0.add(v);
        if (Math.abs(p2.x - p1.x) > Math.abs(p2.y - p1.y)) {
            var ua = (up.x - p1.x) / (p2.x - p1.x);
        } else {
            var ua = (up.y - p1.y) / (p2.y - p1.y);
        }
        if (ua > 0.0 && ua < 1.0) {
            return { ua: ua, up: up };
        }
        return false;
    }

    // Wall is made up of two points
    var Wall = function (p1, p2) {
        this.p1 = p1;
        this.p2 = p2;
    }

    // World object contains many agents and walls and food and stuff
    var util_add_box = function (lst, x, y, w, h) {
        lst.push(new Wall(new Vec(x, y), new Vec(x + w, y)));
        lst.push(new Wall(new Vec(x + w, y), new Vec(x + w, y + h)));
        lst.push(new Wall(new Vec(x + w, y + h), new Vec(x, y + h)));
        lst.push(new Wall(new Vec(x, y + h), new Vec(x, y)));
    }

    // item is circle thing on the floor that agent can interact with (see or eat, etc)
    var Item = function (x, y, type) {
        this.p = new Vec(x, y); // position
        this.type = type;
        this.rad = 10; // default radius
        this.age = 0;
        this.cleanup_ = false;
    }

    var World = function () {
        this.agents = [];
        this.W = canvas.width;
        this.H = canvas.height;

        this.clock = 0;

        // set up walls in the world
        this.walls = [];
        var pad = 10;
        util_add_box(this.walls, pad, pad, this.W - pad * 2, this.H - pad * 2);
        for (var x=100; x< this.W - 100; x+=300) {
            for (var y=100; y < this.H - 100; y += 400) {
                util_add_box(this.walls, x, y, 200, 300); // inner walls
                this.walls.pop();
            }
        }        

        // set up food and poison
        this.items = []
        for (var k = 0; k < num_items; k++) {
            var x = convnetjs.randf(20, this.W - 20);
            var y = convnetjs.randf(20, this.H - 20);
            var t = convnetjs.randi(1, 3); // food or poison (1 and 2)
            var it = new Item(x, y, t);
            this.items.push(it);
        }
    }

    World.prototype = {
        // helper function to get closest colliding walls/items
        stuff_collide_: function (p1, p2, check_walls, check_items, check_agents) {
            var minres = false;

            // collide with walls
            if (check_walls) {
                for (var i = 0, n = this.walls.length; i < n; i++) {
                    var wall = this.walls[i];
                    var res = line_intersect(p1, p2, wall.p1, wall.p2);
                    if (res) {
                        res.type = 0; // 0 is wall
                        if (!minres) { minres = res; }
                        else {
                            // check if its closer
                            if (res.ua < minres.ua) {
                                // if yes replace it
                                minres = res;
                            }
                        }
                    }
                }
            }

            // collide with items
            if (check_items) {
                for (var i = 0, n = this.items.length; i < n; i++) {
                    var it = this.items[i];
                    var res = line_point_intersect(p1, p2, it.p, it.rad);
                    if (res) {
                        res.type = it.type; // store type of item
                        if (!minres) { minres = res; }
                        else {
                            if (res.ua < minres.ua) { minres = res; }
                        }
                    }
                }
            }

            // collide with agents
            if (check_agents) {
                for (var i = 0, n = this.agents.length; i < n; i++) {
                    var it = this.agents[i];
                    // skip self
                    if (it == check_agents) continue;
                    var res = line_point_intersect(p1, p2, it.p, it.rad);
                    if (res) {
                        res.type = 3; // store type of item
                        if (!minres) { minres = res; }
                        else {
                            if (res.ua < minres.ua) { minres = res; }
                        }
                    }
                }
            }

            return minres;
        },
        tick: function () {
            // tick the environment
            this.clock++;

            // fix input to all agents based on environment
            // process eyes
            this.collpoints = [];
            for (var i = 0, n = this.agents.length; i < n; i++) {
                var a = this.agents[i];
                for (var ei = 0, ne = a.eyes.length; ei < ne; ei++) {
                    var e = a.eyes[ei];
                    // we have a line from p to p->eyep
                    var eyep = new Vec(a.p.x + e.max_range * Math.sin(a.angle + e.angle),
                        a.p.y + e.max_range * Math.cos(a.angle + e.angle));
                    var res = this.stuff_collide_(a.p, eyep, true, true, a);
                    if (res) {
                        // eye collided with wall
                        e.sensed_proximity = res.up.dist_from(a.p);
                        e.sensed_type = res.type;
                    } else {
                        e.sensed_proximity = e.max_range;
                        e.sensed_type = -1;
                    }
                }
            }

            // let the agents behave in the world based on their input
            for (var i = 0, n = this.agents.length; i < n; i++) {
                this.agents[i].forward();
            }

            // apply outputs of agents on evironment
            for (var i = 0, n = this.agents.length; i < n; i++) {
                var a = this.agents[i];
                a.op = a.p; // back up old position
                a.oangle = a.angle; // and angle

                // steer the agent according to outputs of wheel velocities
                var v = new Vec(0, a.rad / 2.0);
                v = v.rotate(a.angle + Math.PI / 2);
                var w1p = a.p.add(v); // positions of wheel 1 and 2
                var w2p = a.p.sub(v);
                var vv = a.p.sub(w2p);
                vv = vv.rotate(-a.rot1);
                var vv2 = a.p.sub(w1p);
                vv2 = vv2.rotate(a.rot2);
                var np = w2p.add(vv);
                np.scale(0.5);
                var np2 = w1p.add(vv2);
                np2.scale(0.5);
                a.p = np.add(np2);

                a.angle -= a.rot1;
                if (a.angle < 0) a.angle += 2 * Math.PI;
                a.angle += a.rot2;
                if (a.angle > 2 * Math.PI) a.angle -= 2 * Math.PI;

                // agent is trying to move from p to op. Check walls
                var res = this.stuff_collide_(a.op, a.p, true, false, a);
                if (res) {
                    if (res.type == 3) {
                        a.digestion_signal += -6.0; // ewww agent
                    }
                    // wall collision! reset position
                    a.p = a.op;
                }

                // handle boundary conditions
                if (a.p.x < 0) a.p.x = 0;
                if (a.p.x > this.W) a.p.x = this.W;
                if (a.p.y < 0) a.p.y = 0;
                if (a.p.y > this.H) a.p.y = this.H;
            }

            // tick all items
            var update_items = false;
            for (var i = 0, n = this.items.length; i < n; i++) {
                var it = this.items[i];
                it.age += 1;

                // see if some agent gets lunch
                for (var j = 0, m = this.agents.length; j < m; j++) {
                    var a = this.agents[j];
                    var d = a.p.dist_from(it.p);
                    if (d < it.rad + a.rad) {

                        // wait lets just make sure that this isn't through a wall
                        var rescheck = this.stuff_collide_(a.p, it.p, true, false);
                        if (!rescheck) {
                            // ding! nom nom nom
                            if (it.type === 1) a.digestion_signal += 5.0; // mmm delicious apple
                            if (it.type === 2) a.digestion_signal += -6.0; // ewww poison
                            it.cleanup_ = true;
                            update_items = true;
                            break; // break out of loop, item was consumed
                        }
                    }
                }

                if (it.age > 5000 && this.clock % 100 === 0 && convnetjs.randf(0, 1) < 0.1) {
                    it.cleanup_ = true; // replace this one, has been around too long
                    update_items = true;
                }
            }
            if (update_items) {
                var nt = [];
                for (var i = 0, n = this.items.length; i < n; i++) {
                    var it = this.items[i];
                    if (!it.cleanup_) nt.push(it);
                }
                this.items = nt; // swap
            }
            if (this.items.length < num_items && this.clock % 10 === 0 && convnetjs.randf(0, 1) < item_prob) {
                var newitx = convnetjs.randf(20, this.W - 20);
                var newity = convnetjs.randf(20, this.H - 20);
                var newitt = convnetjs.randi(1, 3); // food or poison (1 and 2)
                var newit = new Item(newitx, newity, newitt);
                this.items.push(newit);
            }

            // agents are given the opportunity to learn based on feedback of their action on environment
            for (var i = 0, n = this.agents.length; i < n; i++) {                 
                this.agents[i].backward();
            }
        }
    }

    // Eye sensor has a maximum range and senses walls
    var Eye = function (angle) {
        this.angle = angle; // angle relative to agent its on
        this.max_range = sensor_range;
        this.sensed_proximity = sensor_range; // what the eye is seeing. will be set in world.tick()
        this.sensed_type = -1; // what does the eye see?
    }

    const num_items = 30 * num_agents;
    const item_prob = 0.25 * num_agents;

    // A single agent
    var Agent = function () {

        // positional information
        this.p = new Vec(Math.random() * (canvas.width - 20) + 10, Math.random() * (canvas.height - 20) + 10);
        this.op = this.p; // old position
        this.angle = 0; // direction facing

        this.actions = [];
        this.actions.push([1, 1]);
        this.actions.push([0.8, 1]);
        this.actions.push([1, 0.8]);
        this.actions.push([0.5, 0]);
        this.actions.push([0, 0.5]);

        // properties
        this.rad = 10;
        this.eyes = [];
        for (var k = 0; k < 9; k++) { this.eyes.push(new Eye((k - 3) * 0.25)); }
                
        // braaain
        //this.brain = new deepqlearn.Brain(this.eyes.length * 3, this.actions.length);
        this.brain = eval(spec)        

        this.reward_bonus = 0.0;
        this.digestion_signal = 0.0;

        // outputs on world
        this.rot1 = 0.0; // rotation speed of 1st wheel
        this.rot2 = 0.0; // rotation speed of 2nd wheel

        this.prevactionix = -1;
    }
    Agent.prototype = {
        forward: function () {
            // in forward pass the agent simply behaves in the environment
            // create input to brain
            var num_eyes = this.eyes.length;
            var input_array = new Array(num_eyes * 4);
            for (var i = 0; i < num_eyes; i++) {
                var e = this.eyes[i];
                input_array[i * 4] = 1.0;
                input_array[i * 4 + 1] = 1.0;
                input_array[i * 4 + 2] = 1.0;
                input_array[i * 4 + 3] = 1.0;
                if (e.sensed_type !== -1) {
                    // sensed_type is 0 for wall, 1 for food and 2 for poison.
                    // lets do a 1-of-k encoding into the input array
                    input_array[i * 4 + e.sensed_type] = e.sensed_proximity / e.max_range; // normalize to [0,1]
                }
            }

            // get action from brain
            var actionix = this.brain.forward(input_array);
            var action = this.actions[actionix];
            this.actionix = actionix; //back this up

            // demultiplex into behavior variables
            this.rot1 = action[0] * 1;
            this.rot2 = action[1] * 1;

            //this.rot1 = 0;
            //this.rot2 = 0;
        },
        backward: function () {
            // in backward pass agent learns.
            // compute reward 
            var proximity_reward = 0.0;
            var num_eyes = this.eyes.length;
            for (var i = 0; i < num_eyes; i++) {
                var e = this.eyes[i];
                // agents dont like to see walls, especially up close
                proximity_reward += e.sensed_type === 0 ? e.sensed_proximity / e.max_range : 1.0;
            }
            proximity_reward = proximity_reward / num_eyes;
            proximity_reward = Math.min(1.0, proximity_reward * 2);

            // agents like to go straight forward
            var forward_reward = 0.0;
            if (this.actionix === 0 && proximity_reward > 0.75) forward_reward = 0.1 * proximity_reward;

            // agents like to eat good things
            var digestion_reward = this.digestion_signal;
            this.digestion_signal = 0.0;

            var reward = proximity_reward + forward_reward + digestion_reward;

            // pass to brain for learning
            this.brain.backward(reward);
        }
    }
    
    // Tick the world
    function tick() {
        w.tick();
        draw_stats()
    }

    function startlearn() {
        w.agents.forEach(agent=>agent.brain.learning = true);
    }
    function stoplearn() {
        w.agents.forEach(agent=>agent.brain.learning = false);
    }

    function shareBrain() {
        let last = w.agents.length-1
        w.agents.forEach((agent,i) => {
            if (i != last) {
                agent.brain.master = w.agents[last].brain
            }            
        })
    }

    function reload() {
        spec = editor.GetText()
        w.agents = _.range(num_agents).map(_ => new Agent()); // this should simply work. I think... ;\
        shareBrain()
    }

    let path = Context.GetScriptFileFullPath(__dirname)    
    let pretrainedPath = path.substr(0,path.lastIndexOf('/'))+'/pretrained.json'

    function loadnet() {
        let j = JSON.parse(Context.ReadStringFromFile(pretrainedPath))
        w.agents.forEach(agent => agent.brain.value_net.fromJSON(j));
        w.agents.forEach(agent => agent.brain.value_net2.fromJSON(j));
        shareBrain()
        stoplearn(); // also stop learning      
    }

    function savenet() {
        var b = w.agents[0].brain;
        b = b.master || b;
        var j = b.value_net.toJSON();
        var t = JSON.stringify(j);
        Context.WriteStringToFile(pretrainedPath,t)
        console.log(pretrainedPath,t)
    }

    var w; // global world object
    var current_interval_id;
    var skipdraw = false;
    function start() {
        w = new World();
        w.agents = _.range(num_agents).map(_ => new Agent()); // this should simply work. I think... ;\
        shareBrain()

        schedule()
    }

    var simspeed = 2;

    function goveryfast() {
      skipdraw = true;
      simspeed = 3;
    }
    function gofast() {
      skipdraw = false;
      simspeed = 2;
    }
    function gonormal() {
      skipdraw = false;
      simspeed = 1;
    }
    function goslow() {
      skipdraw = false;
      simspeed = 0;
    }
    
    function schedule() {
        if (!alive) return
        for (var i=0; i<=simspeed; i++) {
            tick()
        }
        process.nextTick(schedule)
    }

    start()

    let stats = []

    function addStat(key,fn) {
        let o = new JavascriptObject()
        stats.push(o)
        o.key = key
        o.fn = fn
    }
     
    addStat("experience replay size",b => b.experience.length)
    addStat("exploration epsilon",b => b.epsilon)
    addStat('age',b => b.age)
    addStat('average Q-learning loss', b => b.average_loss_window.get_average());
    addStat('smooth-ish reward', b => b.average_reward_window.get_average());

    var reward_graph = new cnnvis.Graph();
    function draw_stats() {
      if(w.clock % 200 === 0) {
        reward_graph.add(w.clock/200, _.meanBy(w.agents,a => a.brain.average_reward_window.get_average()));
      }
    }         
    
    let UMG = require('UMG')
    makeWindow("$convnet",
    {
        SizingRule:'AutoSized',
        Title:'ConvNetJS'
    })(finish => {
        let asset = new SlateBrushAsset
        let style = new JavascriptStyleSet
        style.StyleSetName = 'EditorStyle'
        asset.Brush = style.GetBrush('Sequencer.KeyCircle')
        class TestWidget extends JavascriptWidget {
            OnPaint(context) {
                if (skipdraw) return

                let red = {R:1,A:1}
                let black = {A:1}
                let green = {G:1,A:1}
                let blue = {B:1,A:1}                
                function line(x1,y1,x2,y2,color) {
                    x1 *= scale
                    x2 *= scale
                    y1 *= scale
                    y2 *= scale
                    context.DrawLine({X:x1,Y:y1},{X:x2,Y:y2},color,true)
                }

                function circle(x,y,r,color) {
                    x *= scale
                    y *= scale
                    r *= scale
                    context.DrawBox({X:x-r,Y:y-r},{X:2*r,Y:2*r},asset,color)
                }

                var agents = w.agents;

                for (var i = 0, n = w.walls.length; i < n; i++) {
                    var q = w.walls[i];
                    line(q.p1.x, q.p1.y,q.p2.x, q.p2.y,red);
                }

                // draw agents
                // color agent based on reward it is experiencing at the moment                
                let agent = {R:1,A:1} 
                for (var i = 0, n = agents.length; i < n; i++) {
                    var a = agents[i];

                    var r = Math.floor(a.brain.latest_reward * 200);
                    if (r > 255) r = 255; if (r < 0) r = 0;
                    agent.R = r/255

                    circle(a.op.x, a.op.y, a.rad,agent)                    

                    // draw agents sight
                    for (var ei = 0, ne = a.eyes.length; ei < ne; ei++) {
                        var e = a.eyes[ei];
                        var sr = e.sensed_proximity;
                        let color
                        if (e.sensed_type === -1 || e.sensed_type === 0) {
                            color = black
                        }
                        if (e.sensed_type === 1) { color = red }
                        if (e.sensed_type === 2) { color = green }
                        if (e.sensed_type === 3) { color = blue } 
                        line(a.op.x, a.op.y,a.op.x + sr * Math.sin(a.oangle + e.angle),
                            a.op.y + sr * Math.cos(a.oangle + e.angle),color);
                    }
                }
                
                // draw items
                for (var i = 0, n = w.items.length; i < n; i++) {
                    var it = w.items[i];
                    let color
                    if (it.type === 1) color = red;
                    if (it.type === 2) color = green;
                    circle(it.p.x, it.p.y, it.rad, color);
                }
            }
        }

        class ChartWidget extends JavascriptWidget {
            OnPaint(context) {
                reward_graph.drawSelf(chartcanvas,context)
            }
        }

        let TestWidget_C = require('uclass')()(global,TestWidget)
        let ChartWidget_C = require('uclass')()(global,ChartWidget)

        let font = {
            Size: 10,
            FontObject: Root.GetEngine().SmallFont
        }        

        let Chart = UMG(SizeBox,
            {
                WidthOverride : chartcanvas.width,
                HeightOverride : chartcanvas.height
            },
            UMG(ChartWidget_C,{})
        )
        let Parameters = UMG(SizeBox,
            {
                WidthOverride : chartcanvas.width,                
            },
            UMG(JavascriptListView,
                {
                    'Slot.Size.Rule':'Fill',
                    Columns:[
                        {
                            Id: 'Name',
                            Width: 0.5,
                        },
                        {
                            Id: 'Value',
                            Width: 0.5
                        }
                    ],
                    OnGenerateRowEvent:(item,column) => I(
                        item ? ( 
                            column == 'Name' ?
                            UMG.text({
                                Font:font
                            },item.key) :
                            UMG.text({
                                Font:font,
                                TextDelegate:_ => {
                                    return String(item.fn(w.agents[w.agents.length-1].brain)) 
                                }
                            }) 
                        ) :
                        UMG.text({
                            Font:font
                        },column) 
                    ),
                    $link:elem => {
                        elem.Items = stats
                        elem.RequestListRefresh()
                    }
                })
        )

        let Game = UMG(SizeBox,
            {
                WidthOverride : canvas.width * scale,
                HeightOverride : canvas.height * scale
            },
            UMG(TestWidget_C,{})
        )

        return UMG.div({},
            UMG.span({},
                Game,
                UMG.div({},
                    Parameters,
                    Chart
                )                
            ),
            UMG(Spacer,{'Slot.Size.Rule' : 'Fill'}),
            UMG(SizeBox,
                {
                    HeightOverride:200,
                },
                UMG(MultiLineEditableTextBox, {
                    'slot.size.size-rule': 'Fill',
                    WidgetStyle: style.GetEditableTextBoxStyle('Graph.StateNode.NodeTitleEditableText'),
                    Text: spec,
                    $link: elem => editor = elem                
                })
            ),
            UMG.span({},
                _.map({
                    "Very fast": goveryfast,
                    "Fast": gofast,
                    "Normal": gonormal,
                    "Slow": goslow
                },(v,k) => 
                    UMG(Button,
                        {
                            'Slot.Size.Rule' : 'Fill',
                            WidgetStyle: style.GetButtonStyle("FlatButton.Dark"),
                            OnClicked: v
                        },
                        UMG.text({ Font: font }, k)
                    )
                )
            ),
            UMG.span({},
                _.map({
                    "Load pre-trained net": loadnet,
                    "Save trained net": savenet,
                    "Reload": reload,
                    "Start learn": startlearn,
                    "Stop learn": stoplearn
                },(v,k) => 
                    UMG(Button,
                        {
                            'Slot.Size.Rule' : 'Fill',
                            WidgetStyle: style.GetButtonStyle("FlatButton.Dark"),
                            OnClicked: v
                        },
                        UMG.text({ Font: font }, k)
                    )
                )
            )
        )
    })    

    return function () {
        alive = false
    }
}

function makeWindow(key,opts) {
    const _ = require('lodash')
    const UMG = require('UMG')    

    if (!global[key]) {
        let window
        let container
        let widget = I(
            UMG(JavascriptWindow,_.extend(
                {
                    $link:elem => window = elem
                },opts),
                UMG(SizeBox,{$link:elem => container = elem})
            )
        )
        widget.TakeWidget().AddWindow()

        let prev
        function add(child) {
            if (prev) {
                container.remove_child(prev)
            }
            prev = container.add_child(child(finish))
            process.nextTick(_ => window.BringToFront())
        }

        global[key] = add

        function finish() {
            if (window) {
                window.RequestDestroyWindow()
                window = null
                global[key] = null
            }
        }
    }

    return global[key]
}

module.exports = function() {
    try {
        let bye
        let alive = true
        process.nextTick(_ => {
            if (!alive) return
            bye = main()
        })
        return _ => {
            alive = false
            if (bye) bye()
        }
    } catch (e) {
        console.error(String(e),'got error')
        return _ => null
    }
}
