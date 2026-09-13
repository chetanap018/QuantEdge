// report_charts.js - Advanced Strategy Analysis Visualizations
(function(){
var D = REPORT_DATA;
var M = D.metrics;

// ============================================================
// KPI CARDS
// ============================================================
function fmtMoney(v){return 'Rs '+v.toLocaleString('en-IN',{minimumFractionDigits:2,maximumFractionDigits:2})}
function fmtPct(v){return v.toFixed(2)+'%'}
function cls(v){return v>0?'pos':v<0?'neg':'neu'}

var kpis = [
  ['Net Profit', fmtMoney(M.net_profit), cls(M.net_profit), 'Return: '+fmtPct(M.total_return_pct)],
  ['Win Rate', fmtPct(M.win_rate), cls(M.win_rate-50), M.winning_trades+'W / '+M.losing_trades+'L'],
  ['Profit Factor', M.profit_factor.toFixed(2), cls(M.profit_factor-1), 'Gain/Loss ratio'],
     ['Max Drawdown', fmtPct(M.max_drawdown_pct), cls(5-M.max_drawdown_pct), 'Peak to trough'],
  ['Sharpe Ratio', M.sharpe_ratio.toFixed(2), cls(M.sharpe_ratio), 'Risk-adjusted'],
  ['Total Trades', M.total_trades, 'neu', D.trades_per_day+'/day avg'],
  ['Charges', fmtMoney(M.total_charges), 'neg', fmtPct(M.total_charges/D.capital*100)+' of capital'],
  ['Final Capital', fmtMoney(M.final_capital), cls(M.net_profit), 'From '+fmtMoney(D.capital)],
];
var row = document.getElementById('kpi-row');
kpis.forEach(function(k){
  var d = document.createElement('div'); d.className='kpi';
  d.innerHTML='<div class="k-label">'+k[0]+'</div><div class="k-value '+k[2]+'">'+k[1]+'</div><div class="k-sub">'+k[3]+'</div>';
  row.appendChild(d);
});

// meta
document.getElementById('strat-name').textContent = D.params_used.orb_bars ? 'HERO ORB' : 'Strategy';
document.getElementById('meta-info').textContent = D.meta.symbol+' / '+D.meta.timeframe+' / '+D.meta.bars+' bars / '+D.meta.start.slice(0,10)+' to '+D.meta.end.slice(0,10);

// ============================================================
// HELPER: base chart option
// ============================================================
function baseOpt(extra){
  return Object.assign({
    backgroundColor:'transparent',
    textStyle:{color:'#8b9bb4',fontSize:11},
    grid:{left:55,right:16,top:28,bottom:30},
    tooltip:{trigger:'axis',backgroundColor:'#131c30',borderColor:'#223052',textStyle:{color:'#e5e9f0'}},
    xAxis:{type:'category',axisLine:{lineStyle:{color:'#223052'}},axisLabel:{color:'#64748b',fontSize:10}},
    yAxis:{type:'value',splitLine:{lineStyle:{color:'#141d31'}},axisLabel:{color:'#64748b',fontSize:10}},
  }, extra||{});
}

// ============================================================
// EQUITY CURVE
// ============================================================
echarts.init(document.getElementById('chart-equity')).setOption(baseOpt({
  legend:{data:['Equity'],textStyle:{color:'#8b9bb4',fontSize:11},top:0},
  series:[{
    name:'Equity',type:'line',data:D.equity,showSymbol:false,smooth:true,
    lineStyle:{width:1.8,color:'#4f8cff'},itemStyle:{color:'#4f8cff'},
    areaStyle:{color:{type:'linear',x:0,y:0,x2:0,y2:1,colorStops:[{offset:0,color:'rgba(79,140,255,0.25)'},{offset:1,color:'rgba(79,140,255,0.02)'}]}},
    markLine:{silent:true,data:[{yAxis:D.capital,lineStyle:{color:'#64748b',type:'dashed',width:1}}],label:{show:false}}
  }]
}));

// ============================================================
// DRAWDOWN
// ============================================================
echarts.init(document.getElementById('chart-dd')).setOption(baseOpt({
  series:[{
    name:'Drawdown %',type:'line',data:D.drawdown,showSymbol:false,
    lineStyle:{width:1.5,color:'#ef4444'},itemStyle:{color:'#ef4444'},
    areaStyle:{color:{type:'linear',x:0,y:0,x2:0,y2:1,colorStops:[{offset:0,color:'rgba(239,68,68,0.3)'},{offset:1,color:'rgba(239,68,68,0.02)'}]}}
  }]
}));

// ============================================================
// TRADE WATERFALL (cumulative)
// ============================================================
var cumData = D.trade_diag.map(function(t){return t.cum});
var barColors = D.trade_diag.map(function(t){return t.win?'#10b981':'#ef4444'});
echarts.init(document.getElementById('chart-waterfall')).setOption(baseOpt({
  legend:{data:['Cumulative P&L'],textStyle:{color:'#8b9bb4'},top:0},
  series:[
    {name:'Trade',type:'bar',data:D.trade_diag.map(function(t){return t.net}),itemStyle:{color:function(p){return barColors[p.dataIndex]}},barMaxWidth:12},
    {name:'Cumulative',type:'line',data:cumData,showSymbol:false,lineStyle:{width:2,color:'#ffb020'},itemStyle:{color:'#ffb020'},z:5}
  ]
}));

// ============================================================
// PER-TRADE NET P&L
// ============================================================
echarts.init(document.getElementById('chart-trades')).setOption(baseOpt({
  series:[{
    type:'bar',data:D.trade_diag.map(function(t){return t.net}),
        itemStyle:{color:function(p){return p.value>=0?'#10b981':'#ef4444'}},barMaxWidth:14,
    markLine:{silent:true,data:[{yAxis:0,lineStyle:{color:'#64748b',type:'solid',width:1}}],label:{show:false}}
  }]
}));

// --- Rolling 5-Trade Win Rate ---
echarts.init(document.getElementById('chart-rolling-wr')).setOption(baseOpt({
  legend:{data:['Win Rate %'],textStyle:{color:'#8b9bb4'}},
  yAxis:{axisLabel:{formatter:function(v){return v+'%'}}},
  series:[{name:'Win Rate %',type:'line',data:D.rolling_wr,showSymbol:false,smooth:true,
    lineStyle:{width:2,color:'#4f8cff'},itemStyle:{color:'#4f8cff'},
    markLine:{silent:true,data:[{yAxis:50,lineStyle:{color:'#64748b',type:'dashed'}}]}}]
}));

// --- Rolling 10-Trade Profit Factor ---
echarts.init(document.getElementById('chart-rolling-pf')).setOption(baseOpt({
  legend:{data:['Profit Factor'],textStyle:{color:'#8b9bb4'}},
  yAxis:{axisLabel:{formatter:function(v){return v.toFixed(1)}}},
  series:[{name:'Profit Factor',type:'line',data:D.rolling_pf,showSymbol:false,smooth:true,
    lineStyle:{width:2,color:'#ffb020'},itemStyle:{color:'#ffb020'},
    markLine:{silent:true,data:[{yAxis:1.0,lineStyle:{color:'#64748b',type:'dashed'}}]}}]
}));

// --- Long vs Short Bar ---
echarts.init(document.getElementById('chart-ls-bar')).setOption(baseOpt({
  legend:{data:['Long P&L','Short P&L'],textStyle:{color:'#8b9bb4'}},
  grid:{top:40},
  xAxis:{type:'category',data:['Direction'],show:false},
  yAxis:{type:'value'},
  series:[
    {name:'Long P&L',type:'bar',data:[{value:D.long_pnl,itemStyle:{color:D.long_pnl>=0?'#10b981':'#ef4444'}}],barMaxWidth:40},
    {name:'Short P&L',type:'bar',data:[{value:D.short_pnl,itemStyle:{color:D.short_pnl>=0?'#10b981':'#ef4444'}}],barMaxWidth:40}
  ]
}));

// --- Time-of-Day Performance ---
var tod = D.tod_summary;
echarts.init(document.getElementById('chart-tod')).setOption(baseOpt({
  legend:{data:['Trades','Avg P&L'],textStyle:{color:'#8b9bb4'}},
  xAxis:{type:'category',data:tod.map(function(d){return d.hour}),
         axisLabel:{color:'#64748b',fontSize:9,rotate:45}},
  yAxis:[{type:'value',name:'Trades',position:'left',axisLabel:{color:'#64748b'}},
         {type:'value',name:'Avg P&L',position:'right',axisLabel:{color:'#64748b'}}],
  series:[
    {name:'Trades',type:'bar',data:tod.map(function(d){return d.trades}),itemStyle:{color:'#4f8cff'}},
    {name:'Avg P&L',type:'line',yAxisIndex:1,data:tod.map(function(d){return d.avg_pnl}),
     showSymbol:false,smooth:true,lineStyle:{width:2,color:'#ffb020'},itemStyle:{color:'#ffb020'}}
  ]
}));

// --- Parameter Sensitivity Heatmap ---
document.getElementById('heatmap-desc').textContent =
  'X: orb_bars | Y: stop_atr | Z: Net P&L (at EMA='+D.heatmap_fixed.trend_ema+', hold='+D.heatmap_fixed.min_hold_bars+')';
echarts.init(document.getElementById('chart-heatmap')).setOption({
  backgroundColor:'transparent',
  tooltip:{trigger:'item',formatter:function(p){return 'Net P&L: Rs '+p.value[2].toFixed(2)}},
  grid:{left:60,right:16,top:20,bottom:50},
  xAxis:{type:'category',data:D.heatmap.map(function(d){return d[0]}),
         axisLabel:{color:'#64748b',fontSize:9,rotate:45},name:'orb_bars'},
  yAxis:{type:'category',data:D.heatmap.map(function(d){return d[1]}),
         axisLabel:{color:'#64748b',fontSize:9},name:'stop_atr'},
  series:[{
    name:'Net P&L',type:'heatmap',
    data:D.heatmap.map(function(d){return [d[0],d[1],d[2]]}),
    itemStyle:{color:'#4f8cff'},
    emphasis:{itemStyle:{shadowBlur:8}}
  }]
});

// --- Sweep Top 12 Table ---
var sweepTbody = document.getElementById('sweep-table');
function npClass(v){ return v>0?'pos':(v<0?'neg':'neu'); }
sweepTbody.innerHTML = D.sweep_top12.map(function(r){
  return '<tr><td>'+r.orb_bars+'</td><td>'+r.trend_ema+'</td><td>'+r.min_hold_bars+'</td><td>'+r.stop_atr+'</td><td class="'+npClass(r.net_profit)+'">'+(r.net_profit>=0?'+':'')+r.net_profit.toFixed(2)+'</td><td>'+r.win_rate.toFixed(1)+'</td><td>'+r.pf.toFixed(2)+'</td><td>'+r.trades+'</td><td>'+r.max_dd.toFixed(2)+'</td></tr>';
}).join('');

// --- Sweep Count ---
document.getElementById('sweep-count').textContent = D.sweep_count;

// --- Worst/Best 5 Trades ---
function renderTradeTable(id, data){
  var t = document.getElementById(id);
  t.innerHTML = data.map(function(d){
    var cls = d.pnl>=0?'win':'loss';
    var vcl = d.pnl>=0?'pos':'neg';
    return '<tr><td><span class="pill '+cls+'">'+d.dir+'</span></td><td>'+d.entry+'</td><td class="'+vcl+'">'+d.pnl.toFixed(2)+'</td></tr>';
  }).join('');
}
renderTradeTable('worst-table', D.worst_5);
renderTradeTable('best-table', D.best_5);

// --- All Trades Table ---
var allTbody = document.getElementById('all-trades-table');
allTbody.innerHTML = D.trade_diag.map(function(t){
  function vcl(v){ return v>=0?'pos':'neg'; }
  return '<tr>'+
    '<td>'+t.n+'</td>'+
    '<td><span class="pill '+(t.win?'win':'loss')+'">'+t.dir+'</span></td>'+
    '<td>'+t.entry_t+'</td>'+
    '<td>'+t.exit_t+'</td>'+
    '<td>'+t.entry_p.toFixed(2)+'</td>'+
    '<td>'+t.exit_p.toFixed(2)+'</td>'+
    '<td class="'+vcl(t.gross)+'">'+(t.gross>=0?'+':'')+t.gross.toFixed(2)+'</td>'+
    '<td>'+(t.charges>0?'-':'')+t.charges.toFixed(2)+'</td>'+
    '<td class="'+vcl(t.net)+'">'+(t.net>=0?'+':'')+t.net.toFixed(2)+'</td>'+
    '<td class="'+vcl(t.cum)+'">'+(t.cum>=0?'+':'')+t.cum.toFixed(2)+'</td>'+
    '<td>'+t.held_h+'h</td>'+
  '</tr>';
}).join('');

// --- Recommendations ---
var recDiv = document.getElementById('recommendations');
recDiv.innerHTML = D.recommendations.map(function(r){
  return '<div class="rec">'+
    '<div class="issue">'+r.issue+'</div>'+
    '<div class="detail">'+r.detail+'</div>'+
    '<div class="fix">'+r.fix+'</div>'+
  '</div>';
}).join('');
})();