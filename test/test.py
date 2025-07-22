class WorkingBokehDashboard:
    """
    完全可用的 Bokeh 儀錶板，專為百萬資料點優化
    """

    def __init__(self, df_pandas,df_outlier, webgl_threshold=25000):
        self.df_original = df_pandas.copy()
        self.df_outlier = df_outlier.copy()
        self.webgl_threshold = webgl_threshold
        self.total_points = len(df_pandas)

        # 檢查 Bokeh 版本
        import bokeh
        self.bokeh_version = bokeh.__version__

        # 初始化資料來源
        self.data_sources = {}
        self.data_sources_label = {}
        self.webgl_config = {}

        # 創建資料層級
        self._create_data_layers()

        # 存儲圖表引用
        self.plots = {}

        self.axis_mapping = {
            'X': '低頻',
            'Y': '高頻',
            'Z': '特高頻',
        }

    def _create_data_layers(self):
        """創建多層次資料來源"""

        layer_configs = {
            'ultra_fast': 5000,
            'fast': 15000,
            'medium': 50000,
            'detailed': 150000,
            'full': self.total_points
        }

        for layer_name, max_points in layer_configs.items():
            if max_points >= self.total_points:
                # 使用完整數據
                layer_df = self.df_original.copy()
                layer_outlier = self.df_outlier.copy()
            else:
                # 創建採樣索引
                indices = np.linspace(0, self.total_points - 1, max_points, dtype=int)
                layer_df = self.df_original.iloc[indices].copy()

                # 對異常點數據進行智能採樣
                if len(self.df_outlier) > 0:
                    # 方法1：保留在採樣範圍內的異常點
                    sampled_outlier_mask = self.df_outlier['Time'].isin(layer_df['Time'])
                    # sampled_outlier_mask = self.df_outlier.index.isin(indices)
                    layer_outlier = self.df_outlier[sampled_outlier_mask].copy()

                else:
                    # 如果沒有異常點，創建空的DataFrame
                    layer_outlier = self.df_outlier.copy()  # 保持相同的結構

            # 為主數據添加輔助列
            if len(layer_df) > 0:
                layer_df = layer_df.copy()  # 確保不修改原始數據
                layer_df['time_str'] = layer_df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                layer_df['index'] = range(len(layer_df))
                layer_df['magnitude'] = np.sqrt(
                    layer_df['X_Value'] ** 2 +
                    layer_df['Y_Value'] ** 2 +
                    layer_df['Z_Value'] ** 2
                )

                # 創建主數據源
                self.data_sources[layer_name] = ColumnDataSource(data=dict(
                    x=layer_df['Time'],
                    x_value=layer_df['X_Value'],
                    y_value=layer_df['Y_Value'],
                    z_value=layer_df['Z_Value'],
                    time_str=layer_df['time_str'],
                    index=layer_df['index'],
                    magnitude=layer_df['magnitude']
                ))
            else:
                # 處理空數據的情況
                self.data_sources[layer_name] = ColumnDataSource(data=dict(
                    x=[], x_value=[], y_value=[], z_value=[],
                    time_str=[], index=[], magnitude=[]
                ))

            # 為異常點數據添加輔助列
            if len(layer_outlier) > 0:
                layer_outlier = layer_outlier.copy()  # 確保不修改原始數據
                layer_outlier['time_str'] = layer_outlier['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                layer_outlier['index'] = range(len(layer_outlier))
                layer_outlier['magnitude'] = np.sqrt(
                    layer_outlier['X_Value'] ** 2 +
                    layer_outlier['Y_Value'] ** 2 +
                    layer_outlier['Z_Value'] ** 2
                )

                # 創建異常點數據源
                self.data_sources_label[layer_name] = ColumnDataSource(data=dict(
                    x=layer_outlier['Time'],
                    x_value=layer_outlier['X_Value'],
                    y_value=layer_outlier['Y_Value'],
                    z_value=layer_outlier['Z_Value'],
                    time_str=layer_outlier['time_str'],
                    index=layer_outlier['index'],
                    magnitude=layer_outlier['magnitude'],
                    label = layer_outlier['label'],
                ))
            else:
                # 處理空異常點數據的情況
                self.data_sources_label[layer_name] = ColumnDataSource(data=dict(
                    x=[], x_value=[], y_value=[], z_value=[],
                    time_str=[], index=[], magnitude=[]))

            # WebGL 配置
            self.webgl_config[layer_name] = len(layer_df) > self.webgl_threshold

    def create_plot(self, layer_name, width=2400, height=400, title=""):
        """創建基礎圖表"""

        source = self.data_sources[layer_name]
        label_source = self.data_sources_label[layer_name]
        use_webgl = self.webgl_config[layer_name]

        p = figure(
            width=width,
            height=height,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            x_axis_type='datetime',
            title=title,
            output_backend="webgl" if use_webgl else "canvas"
        )

        # 安全設置滾輪縮放
        try:
            wheel_zoom = p.select_one(WheelZoomTool)
            if wheel_zoom:
                p.toolbar.active_scroll = wheel_zoom
        except:
            pass

        return p, source, label_source

    def create_detailed_chart(self):
        """創建詳細圖表"""

        p, source, label_source = self.create_plot(
            'medium',
            height=500,
            title=f"詳細資料分析 ({len(self.data_sources['medium'].data['x']):,} 點)"
        )

        colors = ['#e74c3c', '#2ecc71', '#3498db']
        labels = list(self.axis_mapping.values())

        # 創建線條渲染器
        line_renderers = []
        circle_renderers = []
        for i, (color, label) in enumerate(zip(colors, labels)):
            key = ['x_value', 'y_value', 'z_value'][i]
            line = p.line('x', key, source=source, line_color=color,
                          line_width=1.5, alpha=0.9, legend_label=label)
            line_renderers.append(line)
            circle = p.circle('x', key, source=label_source,
                              line_color=color, fill_color=color,
                              size=8, line_width=2, alpha=0.9,
                              legend_label=f"{label} 異常點")
            circle_renderers.append(circle)


        # 存儲渲染器供後續使用

        self.plots['detailed_lines']  = line_renderers
        self.plots['detailed_circle'] = circle_renderers

        # 懸停工具
        hover = HoverTool(tooltips=[
            ('時間', '@time_str'),
            (self.axis_mapping['X'], '@x_value{0.0000}'),
            (self.axis_mapping['Y'], '@y_value{0.0000}'),
            (self.axis_mapping['Z'], '@z_value{0.0000}'),
            ('幅度', '@magnitude{0.0000}')
        ])
        p.add_tools(hover)
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"




        return p

    def create_single_axis_charts(self):
        """創建單軸圖表 - 包含異常點支持"""

        plots = {}
        line_renderers = {}
        circle_renderers = {}
        colors = ['#e74c3c', '#2ecc71', '#3498db']
        axes = ['x', 'y', 'z']
        labels = ['低頻', '高頻', '特高頻']

        for axis, color, label in zip(axes, colors, labels):
            p, source, label_source = self.create_plot(
                'detailed',
                height=350,
                title=f"{label} 資料分析"
            )

            # 創建主數據線條
            line = p.line('x', f'{axis}_value', source=source,
                          line_color=color, line_width=2, alpha=0.8,
                          legend_label=f"{label} 數據")
            line_renderers[axis] = line

            # 創建異常點圓圈
            circle = p.circle('x', f'{axis}_value', source=label_source,
                              line_color=color, fill_color=color,
                              size=10, line_width=2, alpha=0.9,
                              legend_label=f"{label} 異常點")
            circle_renderers[axis] = circle

            # 添加懸停工具
            hover = HoverTool(tooltips=[
                ('時間', '@time_str'),
                (f'{label}值', f'@{axis}_value{{0.0000}}'),
                ('幅度', '@magnitude{0.0000}')
            ])
            p.add_tools(hover)

            # 圖例設置
            p.legend.location = "top_left"
            p.legend.click_policy = "hide"

            plots[axis] = p

        # 存儲渲染器供後續使用
        self.plots['single_axis_lines'] = line_renderers
        self.plots['single_axis_circles'] = circle_renderers

        return plots


    def create_advanced_statistics_table(self):
        """創建進階統計表格 - 包含更多統計指標"""
        # 計算進階統計資料
        stats_data = []
        colors = ['🔴', '🟢', '🔵']
        color_codes = ['#e74c3c', '#2ecc71', '#3498db']

        for i, (axis, value) in enumerate(self.axis_mapping.items()):
            col_name = f'{axis}_Value'
            data_series = self.df_original[col_name]

            # 計算百分位數
            q25 = np.percentile(data_series, 25)
            q75 = np.percentile(data_series, 75)
            iqr = q75 - q25

            stats_data.append({
                'axis_icon': colors[i],
                'axis_name': value,
                'count': len(data_series),
                'mean': round(data_series.mean(), 4),
                'std': round(data_series.std(), 4),
                'min': round(data_series.min(), 4),
                'q25': round(q25, 4),
                'median': round(data_series.median(), 4),
                'q75': round(q75, 4),
                'max': round(data_series.max(), 4),
                'iqr': round(iqr, 4),
                'skewness': round(data_series.skew(), 4),
                'kurtosis': round(data_series.kurtosis(), 4),
                'cv': round((data_series.std() / data_series.mean()) * 100, 2) if data_series.mean() != 0 else 0
            })

        # 創建資料來源
        source = ColumnDataSource(data={key: [row[key] for row in stats_data] for key in stats_data[0].keys()})

        # 基本統計表格
        basic_columns = [
            TableColumn(field="axis_icon", title="", width=30, sortable=False),
            TableColumn(field="axis_name", title="軸向", width=80, sortable=False),
            TableColumn(field="count", title="數量", formatter=NumberFormatter(format="0,0"), width=70),
            TableColumn(field="mean", title="平均值", formatter=NumberFormatter(format="0.0000"), width=80),
            TableColumn(field="std", title="標準差", formatter=NumberFormatter(format="0.0000"), width=80),
            TableColumn(field="cv", title="變異係數%", formatter=NumberFormatter(format="0.00"), width=90)
        ]

        basic_table = DataTable(
            source=source,
            columns=basic_columns,
            width=440,
            height=150,
            index_position=None,
            sortable=True,
            sizing_mode="fixed"
        )

        # 分佈統計表格
        distribution_columns = [
            TableColumn(field="axis_icon", title="", width=30, sortable=False),
            TableColumn(field="min", title="最小值", formatter=NumberFormatter(format="0.0000"), width=70),
            TableColumn(field="q25", title="Q1", formatter=NumberFormatter(format="0.0000"), width=70),
            TableColumn(field="median", title="中位數", formatter=NumberFormatter(format="0.0000"), width=70),
            TableColumn(field="q75", title="Q3", formatter=NumberFormatter(format="0.0000"), width=70),
            TableColumn(field="max", title="最大值", formatter=NumberFormatter(format="0.0000"), width=70),
            TableColumn(field="iqr", title="IQR", formatter=NumberFormatter(format="0.0000"), width=70)
        ]

        distribution_table = DataTable(
            source=source,
            columns=distribution_columns,
            width=450,
            height=150,
            index_position=None,
            sortable=True,
            sizing_mode="fixed"
        )

        # 形狀統計表格
        shape_columns = [
            TableColumn(field="axis_icon", title="", width=30, sortable=False),
            TableColumn(field="axis_name", title="軸向", width=80, sortable=False),
            TableColumn(field="skewness", title="偏度", formatter=NumberFormatter(format="0.0000"), width=80),
            TableColumn(field="kurtosis", title="峰度", formatter=NumberFormatter(format="0.0000"), width=80)
        ]

        shape_table = DataTable(
            source=source,
            columns=shape_columns,
            width=450,
            height=150,
            index_position=None,
            sortable=True,
            sizing_mode="fixed"
        )

        # 創建標題組件
        basic_title = Div(
            text="""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 8px; border-radius: 8px; text-align: center;">
                <strong>📈 基本統計</strong>
            </div>
            """,
            width=450, height=70
        )

        distribution_title = Div(
            text="""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        color: white; padding: 8px; border-radius: 8px; text-align: center;">
                <strong>📊 分佈統計</strong>
            </div>
            """,
            width=450, height=70
        )

        shape_title = Div(
            text="""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        color: white; padding: 8px; border-radius: 8px; text-align: center;">
                <strong>📐 形狀統計</strong>
            </div>
            """,
            width=450, height=70
        )

        # 組合佈局
        tables_row = bokeh_row(
            column(basic_title, basic_table, width=440),
            Div(text="", width=10, height=20),  # 間距
            column(distribution_title, distribution_table, width=450),
            Div(text="", width=10, height=20),  # 間距
            column(shape_title, shape_table, width=270),
            sizing_mode="fixed"
        )
        # 總標題
        # 總標題 - 修正：移除 style 參數，將樣式直接寫在 HTML 中
        main_title = Div(
            text="""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 15px; border-radius: 10px; text-align: center; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15); margin-bottom: 15px;">
                <h3 style="margin: 0; font-weight: 600; font-size: 20px;">
                    📊 完整統計分析報告
                </h3>
            </div>
            """,
            width=1200, height=80  # 調整高度以適應內容
        )

        return column(main_title, tables_row, sizing_mode="scale_width")

    def create_data_outlier_table(self):
        source = self.data_sources_label['full']  # 使用中等精度資料

        columns = [
            TableColumn(field="time_str", title="時間", width=150),
            TableColumn(field="x_value", title=self.axis_mapping['X'],
                        formatter=NumberFormatter(format="0.0000"), width=100),
            TableColumn(field="y_value", title=self.axis_mapping['Y'],
                        formatter=NumberFormatter(format="0.0000"), width=100),
            TableColumn(field="z_value", title=self.axis_mapping['Z'],
                        formatter=NumberFormatter(format="0.0000"), width=100),
            TableColumn(field="label", title="異常軸",
                        formatter=StringFormatter(), width=100),
        ]

        return DataTable(
            source=source,
            columns=columns,
            width=2400,
            height=400,
            sizing_mode="scale_width"
        )


    def create_data_table(self):
        """創建資料表格"""

        source = self.data_sources['medium']  # 使用中等精度資料

        columns = [
            TableColumn(field="time_str", title="時間", width=150),
            TableColumn(field="x_value", title=self.axis_mapping['X'],
                        formatter=NumberFormatter(format="0.0000"), width=100),
            TableColumn(field="y_value", title=self.axis_mapping['Y'],
                        formatter=NumberFormatter(format="0.0000"), width=100),
            TableColumn(field="z_value", title=self.axis_mapping['Z'],
                        formatter=NumberFormatter(format="0.0000"), width=100),
            TableColumn(field="magnitude", title="幅度",
                        formatter=NumberFormatter(format="0.0000"), width=100)
        ]

        return DataTable(
            source=source,
            columns=columns,
            width=2400,
            height=400,
            sizing_mode="scale_width"
        )

    def detailed_plot_bind_data_outlier_table(self, detailed_plot, data_outlier_table):
        """
        綁定詳細圖表與異常點表格，實現視野變化時動態更新表格內容
        只顯示當前視野範圍內的異常點
        """

        # 創建用於視野內異常點的新數據源
        viewport_outlier_source = ColumnDataSource(data=dict(
            time_str=[],
            x_value=[],
            y_value=[],
            z_value=[],
            label=[]
        ))

        # 更新表格的數據源為視野數據源
        data_outlier_table.source = viewport_outlier_source

        # 存儲視野數據源的引用
        self.viewport_outlier_source = viewport_outlier_source

        # 創建視野變化回調函數
        viewport_callback = CustomJS(args=dict(
            full_outlier_source=self.data_sources_label['full'],
            viewport_outlier_source=viewport_outlier_source,
            x_range=detailed_plot.x_range,
            y_range=detailed_plot.y_range
        ), code="""
            // 獲取當前視野範圍
            const x_start = x_range.start;
            const x_end = x_range.end;
            const y_start = y_range.start;
            const y_end = y_range.end;

            // 獲取完整異常點數據
            const full_data = full_outlier_source.data;
            const full_x = full_data['x'];
            const full_x_value = full_data['x_value'];
            const full_y_value = full_data['y_value'];
            const full_z_value = full_data['z_value'];
            const full_time_str = full_data['time_str'];
            const full_label = full_data['label'];

            // 篩選視野內的異常點
            const viewport_time_str = [];
            const viewport_x_value = [];
            const viewport_y_value = [];
            const viewport_z_value = [];
            const viewport_label = [];

            for (let i = 0; i < full_x.length; i++) {
                // 將時間轉換為毫秒數進行比較
                const point_time = new Date(full_x[i]).getTime();
                const x_start_ms = new Date(x_start).getTime();
                const x_end_ms = new Date(x_end).getTime();

                // 檢查時間是否在視野範圍內
                if (point_time >= x_start_ms && point_time <= x_end_ms) {
                    // 根據異常軸標籤檢查對應的Y值是否在視野範圍內
                    let y_value_to_check;
                    const label = full_label[i];

                    if (label === '低頻') {
                        y_value_to_check = full_x_value[i];
                    } else if (label === '高頻') {
                        y_value_to_check = full_y_value[i];
                    } else if (label === '特高頻') {
                        y_value_to_check = full_z_value[i];
                    } else {
                        // 如果標籤不明確，檢查是否有任何軸的值在範圍內
                        y_value_to_check = full_x_value[i]; // 默認檢查X軸
                        if (!(full_x_value[i] >= y_start && full_x_value[i] <= y_end)) {
                            if (full_y_value[i] >= y_start && full_y_value[i] <= y_end) {
                                y_value_to_check = full_y_value[i];
                            } else if (full_z_value[i] >= y_start && full_z_value[i] <= y_end) {
                                y_value_to_check = full_z_value[i];
                            }
                        }
                    }

                    // 檢查Y值是否在視野範圍內
                    if (y_value_to_check >= y_start && y_value_to_check <= y_end) {
                        viewport_time_str.push(full_time_str[i]);
                        viewport_x_value.push(full_x_value[i]);
                        viewport_y_value.push(full_y_value[i]);
                        viewport_z_value.push(full_z_value[i]);
                        viewport_label.push(full_label[i]);
                    }
                }
            }

            // 更新視野表格數據
            viewport_outlier_source.data = {
                'time_str': viewport_time_str,
                'x_value': viewport_x_value,
                'y_value': viewport_y_value,
                'z_value': viewport_z_value,
                'label': viewport_label
            };

            // 觸發更新
            viewport_outlier_source.change.emit();

            // 在控制台輸出調試信息
            console.log(`視野範圍: X(${new Date(x_start).toLocaleString()} - ${new Date(x_end).toLocaleString()}), Y(${y_start.toFixed(2)} - ${y_end.toFixed(2)})`);
            console.log(`視野內異常點數量: ${viewport_time_str.length}`);
        """)

        # 將回調綁定到視野變化事件
        detailed_plot.x_range.js_on_change('start', viewport_callback)
        detailed_plot.x_range.js_on_change('end', viewport_callback)
        detailed_plot.y_range.js_on_change('start', viewport_callback)
        detailed_plot.y_range.js_on_change('end', viewport_callback)

        # 初始化視野表格（顯示所有異常點）
        initial_viewport_callback = CustomJS(args=dict(
            full_outlier_source=self.data_sources_label['full'],
            viewport_outlier_source=viewport_outlier_source
        ), code="""
            const full_data = full_outlier_source.data;
            viewport_outlier_source.data = {
                'time_str': full_data['time_str'],
                'x_value': full_data['x_value'],
                'y_value': full_data['y_value'],
                'z_value': full_data['z_value'],
                'label': full_data['label']
            };
            viewport_outlier_source.change.emit();
        """)

        # 在創建完成後執行初始化
        from bokeh.io import curdoc
        if curdoc().session_context:
            curdoc().add_next_tick_callback(lambda: exec(initial_viewport_callback.code))

    def create_controls(self):
        """
        修改後的控制台創建函數，包含視野表格匯出功能
        """

        # 前面的控制元件創建代碼保持不變...
        data_select = Select(
            title="🎯 資料精度",
            value="medium",
            options=[
                ("ultra_fast", "⚡ 超快速 (5K)"),
                ("fast", "🚀 快速 (15K)"),
                ("medium", "⚖️ 中等 (50K)"),
                ("detailed", "🔍 詳細 (150K)"),
                ("full", "💎 完整資料")
            ],
            width=220,
            height=60
        )

        axis_check = CheckboxButtonGroup(
            labels=["📊 低頻軸", "📈 高頻軸", "📉 特高頻軸"],
            active=[0, 1, 2],
            width=600,
            height=60,
            margin=50,
        )

        outlier_toggle = Toggle(
            label="🚨 顯示異常點",
            active=True,
            width=150,
            height=60,
            button_type="warning"
        )

        # 修改後的匯出按鈕
        export_all_outliers = Button(
            label="📥 匯出全部異常點",
            width=180,
            height=60,
            button_type="success"
        )

        # 修改這個按鈕的功能 - 匯出視野內的異常點
        export_viewport_outliers = Button(
            label="🔍 匯出視野異常點",
            width=180,
            height=60,
            button_type="primary"
        )

        export_all_data = Button(
            label="📊 匯出全部資料",
            width=180,
            height=60,
            button_type="default"
        )

        # 狀態顯示
        status_div = Div(
            text="""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 10px; border-radius: 8px; text-align: center;">
                <strong>📊 當前狀態：中等精度 | 🔍 顯示軸：低頻, 高頻, 特高頻 | 🚨 異常點：開啟</strong>
            </div>
            """,
            width=800,
            height=60
        )

        export_status_div = Div(
            text="""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                        color: white; padding: 8px; border-radius: 6px; text-align: center; font-size: 14px;">
                <strong>💾 匯出功能已就緒</strong>
            </div>
            """,
            width=600,
            height=40
        )

        # JavaScript 函數保持不變
        csv_export_js = """
        function downloadCSV(data, filename) {
            const headers = ['時間', '低頻值', '高頻值', '特高頻值', '異常軸'];
            const csvRows = [];
            csvRows.push(headers.join(','));

            for (let i = 0; i < data.time_str.length; i++) {
                const row = [
                    `"${data.time_str[i]}"`,
                    data.x_value[i].toFixed(4),
                    data.y_value[i].toFixed(4),
                    data.z_value[i].toFixed(4),
                    `"${data.label[i]}"`
                ];
                csvRows.push(row.join(','));
            }

            const csvContent = csvRows.join('\\n');
            const BOM = '\\uFEFF';
            const blob = new Blob([BOM + csvContent], { type: 'text/csv;charset=utf-8;' });

            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', filename);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            const timestamp = new Date().toLocaleString('zh-TW');
            window.export_status_div.text = `
                <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                            color: white; padding: 8px; border-radius: 6px; text-align: center; font-size: 14px;">
                    <strong>✅ 成功匯出 ${filename} (${timestamp})</strong>
                </div>
            `;

            setTimeout(() => {
                window.export_status_div.text = `
                    <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                                color: white; padding: 8px; border-radius: 6px; text-align: center; font-size: 14px;">
                        <strong>💾 匯出功能已就緒</strong>
                    </div>
                `;
            }, 3000);
        }
        """

        # 匯出視野內異常點的回調函數（新增）
        export_viewport_callback = CustomJS(args=dict(
            viewport_outlier_source=self.viewport_outlier_source if hasattr(self, 'viewport_outlier_source') else None,
            export_status_div=export_status_div
        ), code=f"""
            {csv_export_js}

            window.export_status_div = export_status_div;

            if (!viewport_outlier_source) {{
                export_status_div.text = `
                    <div style="background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%); 
                                color: white; padding: 8px; border-radius: 6px; text-align: center; font-size: 14px;">
                        <strong>⚠️ 視野表格未初始化</strong>
                    </div>
                `;
                return;
            }}

            const viewport_data = viewport_outlier_source.data;
            const viewport_count = viewport_data.time_str.length;

            if (viewport_count === 0) {{
                export_status_div.text = `
                    <div style="background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%); 
                                color: white; padding: 8px; border-radius: 6px; text-align: center; font-size: 14px;">
                        <strong>⚠️ 當前視野沒有異常點</strong>
                    </div>
                `;
                setTimeout(() => {{
                    export_status_div.text = `
                        <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                                    color: white; padding: 8px; border-radius: 6px; text-align: center; font-size: 14px;">
                            <strong>💾 匯出功能已就緒</strong>
                        </div>
                    `;
                }}, 3000);
                return;
            }}

            const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
            const filename = `視野內異常點數據_${{timestamp}}.csv`;

            export_status_div.text = `
                <div style="background: linear-gradient(135deg, #007bff 0%, #6610f2 100%); 
                            color: white; padding: 8px; border-radius: 6px; text-align: center; font-size: 14px;">
                    <strong>⏳ 正在匯出視野內 ${{viewport_count}} 個異常點...</strong>
                </div>
            `;

            setTimeout(() => {{
                downloadCSV(viewport_data, filename);
            }}, 500);
        """)

        # 匯出全部異常點回調（保持不變）
        export_all_callback = CustomJS(args=dict(
            label_sources=self.data_sources_label,
            export_status_div=export_status_div
        ), code=f"""
            {csv_export_js}
            window.export_status_div = export_status_div;

            const full_outlier_data = label_sources['full'];
            const total_outliers = full_outlier_data.data.x.length;

            if (total_outliers === 0) {{
                export_status_div.text = `
                    <div style="background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%); 
                                color: white; padding: 8px; border-radius: 6px; text-align: center; font-size: 14px;">
                        <strong>⚠️ 沒有異常點數據可匯出</strong>
                    </div>
                `;
                setTimeout(() => {{
                    export_status_div.text = `
                        <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                                    color: white; padding: 8px; border-radius: 6px; text-align: center; font-size: 14px;">
                            <strong>💾 匯出功能已就緒</strong>
                        </div>
                    `;
                }}, 3000);
                return;
            }}

            const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
            const filename = `全部異常點數據_${{timestamp}}.csv`;

            export_status_div.text = `
                <div style="background: linear-gradient(135deg, #007bff 0%, #6610f2 100%); 
                            color: white; padding: 8px; border-radius: 6px; text-align: center; font-size: 14px;">
                    <strong>⏳ 正在匯出 ${{total_outliers}} 個異常點...</strong>
                </div>
            `;

            setTimeout(() => {{
                downloadCSV(full_outlier_data.data, filename);
            }}, 500);
        """)

        # 其他回調函數保持不變...（省略以節省空間）

        # 綁定回調函數
        export_all_outliers.js_on_click(export_all_callback)
        export_viewport_outliers.js_on_click(export_viewport_callback)  # 綁定新的視野匯出功能

        # 存儲控制元件引用
        self.controls = {
            'data_select': data_select,
            'axis_check': axis_check,
            'outlier_toggle': outlier_toggle,
            'export_all_outliers': export_all_outliers,
            'export_viewport_outliers': export_viewport_outliers,  # 更新引用名稱
            'export_all_data': export_all_data,
            'status_div': status_div,
            'export_status_div': export_status_div
        }

        # 創建匯出按鈕區域（更新按鈕）
        export_buttons_section = column(
            Div(text="<h4 style='margin: 5px 0; color: #495057;'>💾 資料匯出</h4>",
                width=600, height=25),
            row(
                export_all_outliers,
                Div(text="", width=10, height=20),
                export_viewport_outliers,  # 使用新的視野匯出按鈕
                Div(text="", width=10, height=20),
                export_all_data,
                sizing_mode="scale_width"
            ),
            export_status_div,
            sizing_mode="scale_width"
        )

        # 其餘佈局代碼保持不變...
        controls_row1 = row(
            data_select,
            Div(text="", width=20, height=20),
            axis_check,
            Div(text="", width=20, height=20),
            outlier_toggle,
            sizing_mode="scale_width"
        )

        controls_section = column(
            status_div,
            Div(text="", width=20, height=10),
            row(controls_row1, export_buttons_section),
            Div(text="", width=20, height=15),
            sizing_mode="scale_width"
        )

        return controls_section


    def create_dashboard(self, name, output_file_path="working_bokeh_dashboard.html"):
        """創建完整儀錶板"""

        # 創建所有組件
        detailed_plot = self.create_detailed_chart()
        single_plots = self.create_single_axis_charts()
        stats_table = self.create_advanced_statistics_table()
        data_table = self.create_data_table()
        data_outlier_table = self.create_data_outlier_table()


        # 存儲表格引用以便 JavaScript 控制
        self.plots['data_table'] = data_table
        self.plots['data_outlier_table'] = data_outlier_table

        self.detailed_plot_bind_data_outlier_table(detailed_plot, data_table)

        # 創建控制台（必須在所有圖表創建之後）
        controls = self.create_controls()


        # 其餘代碼保持不變...
        # 標題
        title_html = f"""
                    <style>
                        .full-width-header {{
                            width: 100vw;
                            margin-left: 0;
                            padding: 20px 0;
                            text-align: center;
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                            box-sizing: border-box;
                        }}
                        .header-content {{
                            max-width: 2400;
                            margin: 0 auto;
                            padding: 0 20px;
                        }}
                    </style>
                    <div class="full-width-header">
                        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 600;">
                                🚀 Power Guard History Data 資料儀錶板
                        </h1>
                        <h3 style="margin: 0; font-size: 2.5rem; font-weight: 600;">
                                    {name}
                        </h3>
                        <div class="header-content">

                        </div>
                    </div>
                    """

        title_div = Div(text=title_html, width=2400, height=120)

        # 控制台區域
        control_section = column(
            controls,
            Div(text="<hr>", width=2400, height=20)
        )

        # 其餘佈局代碼保持不變...
        # 主要圖表區域
        main_charts = column(
            Div(text="<h3>📊 主要圖表</h3>", width=2400, height=40),
            detailed_plot,
            Div(text="<hr>", width=2400, height=20)
        )

        # 單軸分析區域
        single_axis_section = column(
            Div(text="<h3>📈 單軸分析</h3>", width=2400, height=40),
            single_plots['x'],
            single_plots['y'],
            single_plots['z'],
            Div(text="<hr>", width=2400, height=20)
        )

        # 高級分析區域
        advanced_section = column(
            Div(text="<h3>🎯 高級分析</h3>", width=2400, height=40),
            stats_table,
            Div(text="<hr>", width=2400, height=20)
        )

        # 原始資料區域
        data_section = column(
            Div(text="<h3>📋 原始資料</h3>", width=2400, height=40),
            data_table
        )

        # 組合最終佈局
        layout = column(
            title_div,
            control_section,
            bokeh_row(main_charts,data_outlier_table) ,
            single_axis_section,
            advanced_section,
            data_section,
            sizing_mode="scale_width"
        )

        # 輸出
        output_file(output_file_path)
        save(layout)

        return output_file_path
