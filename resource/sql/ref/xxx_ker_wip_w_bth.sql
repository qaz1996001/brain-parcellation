CREATE DATABASE  IF NOT EXISTS `mes_omi` /*!40100 DEFAULT CHARACTER SET utf8 */;
USE `mes_omi`;
-- MySQL dump 10.13  Distrib 5.7.12, for Win32 (AMD64)
--
-- Host: 127.0.0.1    Database: mes_omi
-- ------------------------------------------------------
-- Server version	5.5.5-10.2.7-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `xxx_ker_wip_w_bth`
--

DROP TABLE IF EXISTS `xxx_ker_wip_w_bth`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `xxx_ker_wip_w_bth` (
  `report_time` datetime NOT NULL COMMENT 'tomorrow am 08:00 is real time data,others is history  -- 2017/06/26 lkchena\n輪三班制 We work on 3 shifts. \n上早班 on the day shift \n上夜班 on the night shift \n上大夜班 on the overnight shift\n',
  `cate` varchar(6) NOT NULL COMMENT 'category:\nRT : real-time\nYES : yesterday\nS1: day shift\nS2: night shift\nS3: over-night shift',
  `lot_id` varchar(16) NOT NULL COMMENT 'lot 編碼加上屬性別 –> need confirm with Auronal\n\nlot_id:  A1234567.99\n第一碼: A --> 廠別\n第二碼: 6,7,8: SHR\n 1,2,3,4: product\n 5,9: RD\n尾碼: 99 --> 00: 母批 其他:子批 	\n\n--> 2017/07/20\n尾碼: 99  --> 00: 母批 其他: for cassette use',
  `lot_id_p1` varchar(16) NOT NULL COMMENT '2020/03/04 lkchena\n\n1. we design: lot / wafer use the same field with different table name\n2. ker_wip_bt: lot_id = lot_id_p\n3. ker_wip_w_bt: lot_id = box_id, lot_id_p1 = box parent = lot_id\n4. ker_wip_w2_bt: lot_id = wafer_id,   lot_id_p2= wafer parent = box_id,  lot_id_p1 = box parent = lot_id',
  `lot_id_p2` varchar(16) DEFAULT NULL COMMENT '2020/03/04 lkchena\n\n1. we design: lot / wafer use the same field with different table name\n2. ker_wip_bt: lot_id = lot_id_p\n3. ker_wip_w_bt: lot_id = box_id, lot_id_p1 = box parent = lot_id\n4. ker_wip_w2_bt: lot_id = wafer_id,   lot_id_p2= wafer parent = box_id,  lot_id_p1 = box parent = lot_id',
  `cart_no` varchar(16) DEFAULT NULL COMMENT 'extend length to 12 -- 2020/03/02 lkchena\nwait design naming rule\n',
  `box_no` varchar(16) DEFAULT NULL,
  `lot_size` int(11) NOT NULL DEFAULT 0 COMMENT '幾乎等於 cassette 數量, 不是全部都是 40, 因為有可能換班or料用完, 做不滿 40, 此時就會輸入數量 >> 對應的就是 cassette 數量變少(不是滿批)',
  `lot_size_spec` int(11) NOT NULL DEFAULT 0,
  `box_pcs_spec` int(11) NOT NULL DEFAULT 0 COMMENT '一個 cassette 可以裝多少個 part, 應該定義在 part table, 而不是 carrier 上\n<-- get data from wip table(from part table)',
  `box_spec` int(11) NOT NULL DEFAULT 0,
  `box_cnt` int(11) NOT NULL DEFAULT 0,
  `s` varchar(1) DEFAULT NULL COMMENT 'R,Q,H,B,S,E,K,P\nB: Backup(協力廠)\nS: plan start(plan stb)(not process, still in ERP db)\nK: skip(manual handle)(auto無法一次到位)\nP: by-pass step\n',
  `pri` int(11) DEFAULT NULL COMMENT 'priority: 1~999\nSmaller has priority.\n\n1~10: SHR\n11~100: Important customer',
  `part_id` varchar(24) NOT NULL,
  `part_raw` varchar(16) DEFAULT NULL COMMENT '產品的原料: 旭宏 現場 習慣看 原料的編號 -- 2017/07/20 lkchena',
  `route_id` varchar(24) NOT NULL DEFAULT 'ROUTE-ZZZ-01' COMMENT 'for user not need to create each part / step record -- 2020/03/26 lkchena',
  `raw_2d_code` varchar(16) DEFAULT NULL COMMENT '2017/07/27 保來得: 鐵粉原料有 2d-code,而且用不完,還要繳回庫房\n\n因為無法跟著 part table 走, 只好 當 wafer_start 時(成型站), 寫入',
  `ope_no` varchar(7) NOT NULL COMMENT '2020/03/17 lkchena: for option flow, extend length to 7',
  `ope_name` varchar(36) DEFAULT NULL,
  `stage_id` varchar(16) DEFAULT NULL,
  `stage_name` varchar(36) DEFAULT NULL,
  `stage_order` varchar(3) DEFAULT NULL COMMENT 'ope_no 的前三碼(main step, not sub steps)',
  `ope_cate` varchar(12) DEFAULT NULL COMMENT 'operation category: track_in,proc_start,proc_end,track_out',
  `extra_step` int(11) NOT NULL DEFAULT 0 COMMENT '0: default flow step 1: extra - step   -- for option flow\n2020/03/22 lkchena',
  `claim_time` datetime DEFAULT NULL,
  `claim_user` varchar(16) DEFAULT NULL,
  `tool_grp_id` varchar(24) DEFAULT NULL,
  `tool_grp` varchar(32) DEFAULT NULL,
  `ws_type` varchar(12) DEFAULT '0' COMMENT 'change to use string, compatible with aruroal 2020/03/29 lkchena\n2020/03/17 lkchena\n',
  `ws_func` varchar(24) DEFAULT NULL,
  `tool_type1` varchar(12) NOT NULL DEFAULT '0' COMMENT 'for multi-step in one step, ex: 燒結乾振防鏽\n2020/03/22 lkchena',
  `tool_type2` varchar(12) NOT NULL DEFAULT '0' COMMENT 'for multi-step in one step, ex: 燒結乾振防鏽\n2020/03/22 lkchena',
  `tool_type3` varchar(12) NOT NULL DEFAULT '0' COMMENT 'for multi-step in one step, ex: 燒結乾振防鏽\n2020/03/22 lkchena',
  `in_out` int(11) NOT NULL DEFAULT 0 COMMENT 'default: 0: 廠內  1: 外注 1x: maybe A,B factory\n2020/03/25 lkchena',
  `tool_func` varchar(24) DEFAULT NULL,
  `er` varchar(32) DEFAULT NULL COMMENT 'equipment recipe:  2017/07/20 lkchena\n\nwe don''t use recipe group, recipe_id ......, cause the process is too sample\n',
  `area_id` varchar(12) DEFAULT NULL,
  `area_name` varchar(24) DEFAULT NULL,
  `pos_id` varchar(12) DEFAULT NULL COMMENT '2020/03/22 lkchena',
  `pos_desc` varchar(36) DEFAULT NULL COMMENT '2020/03/22 lkchena',
  `tool_id` varchar(12) DEFAULT NULL COMMENT 'length extend to 12 for sintering "before/after" omi -- 2020/03/18 lkchena',
  `tag_track_io` varchar(16) DEFAULT NULL COMMENT 'for sintering product  track in/out mark in  the plate -- 2020/03/22 lkchena\nrfid_tag_b is tag number ',
  `proc_time` int(11) DEFAULT NULL,
  `cust_id` varchar(12) DEFAULT NULL,
  `cust_name` varchar(36) DEFAULT NULL,
  `order_no` varchar(16) DEFAULT NULL COMMENT '2017/11/21 lkchena: additon for tracking order',
  `mfg_no` varchar(16) DEFAULT NULL,
  `rd_flag` int(11) NOT NULL DEFAULT 0 COMMENT '... leak the field -- 2020/03/20 lkchena',
  `b_vendor_id` varchar(6) DEFAULT NULL COMMENT '當 s = "B" -- backup 時, 寫入 vendor_id',
  `b_vendor_name` varchar(24) DEFAULT NULL,
  `lot_note` varchar(64) DEFAULT NULL COMMENT 'operate note information in lot',
  `lot_memo` varchar(64) DEFAULT NULL COMMENT 'system log information in lot',
  `track_in_time` datetime DEFAULT NULL,
  `track_out_time` datetime DEFAULT NULL,
  `proc_start_time` datetime DEFAULT NULL,
  `proc_end_time` datetime DEFAULT NULL,
  `stb_plan` datetime DEFAULT NULL,
  `stb_real` datetime DEFAULT NULL,
  `mold_id` varchar(16) DEFAULT NULL COMMENT 'mold_id will use at spc analysis - 2017/11/23 lkchena',
  `powder_id` varchar(24) DEFAULT NULL,
  `powder_type` varchar(16) DEFAULT NULL,
  `cnt_plan` int(11) NOT NULL DEFAULT 0,
  `cnt_cur` int(11) NOT NULL DEFAULT 0 COMMENT 'current count',
  `cnt_act` int(11) NOT NULL DEFAULT 0 COMMENT '實際驗收-pass qa -- 2020/02/14 lkchena',
  `cnt_ng` int(11) NOT NULL DEFAULT 0 COMMENT 'ng count - 2018/10/23 lkchena',
  `cnt_qc` int(11) NOT NULL DEFAULT 0,
  `cnt_test` int(11) NOT NULL DEFAULT 0 COMMENT '調機 -- 2020/02/14 lkchena',
  `cnt_tool` int(11) NOT NULL DEFAULT 0 COMMENT 'tool amount -- 2020/04/15 lkchena',
  `cnt_empty` int(11) NOT NULL DEFAULT 0 COMMENT '空打 ?',
  `split_from` varchar(64) DEFAULT NULL COMMENT '2020/04/06 lkchena\nfor sorter s/m key field',
  `merge_from` varchar(64) DEFAULT NULL COMMENT '2020/04/06 lkchena\nfor sorter s/m key field',
  `full_num` varchar(1024) DEFAULT NULL COMMENT 'see ker_wip_bt definiton -- 2020/03/04 lkchena',
  `full_code` varchar(1024) DEFAULT NULL COMMENT 'see ker_wip_bt definiton -- 2020/03/04 lkchena',
  `full_time` varchar(2048) DEFAULT NULL COMMENT 'see ker_wip_bt definiton -- 2020/03/04 lkchena',
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`lot_id`,`report_time`,`cate`),
  KEY `idx_cart_id` (`cart_no`) COMMENT 'for sorter merge 2020/03/29 lkchena',
  KEY `idx_lot_p_cart` (`lot_id_p1`,`cart_no`) COMMENT 'for sorter merge query data -- 2020/03/29 lkchena'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AVG_ROW_LENGTH=256 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `xxx_ker_wip_w_bth`
--

LOCK TABLES `xxx_ker_wip_w_bth` WRITE;
/*!40000 ALTER TABLE `xxx_ker_wip_w_bth` DISABLE KEYS */;
/*!40000 ALTER TABLE `xxx_ker_wip_w_bth` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:22
