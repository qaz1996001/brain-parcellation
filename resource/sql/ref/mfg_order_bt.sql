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
-- Table structure for table `mfg_order_bt`
--

DROP TABLE IF EXISTS `mfg_order_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `mfg_order_bt` (
  `mfg_no` varchar(16) NOT NULL COMMENT '??? why has mfg_no --> Trinity has the mfg_no, why ??? 2019/12/31 lkchena',
  `order_no` varchar(16) DEFAULT NULL COMMENT 'customer order - 2019/12/08 lkchena\nmaybe null(?)',
  `rd_flag` int(11) NOT NULL DEFAULT 0 COMMENT '0: prod 1: test run 2: rd \n2019/02/03 lkchena',
  `status1` int(11) NOT NULL DEFAULT 0 COMMENT 'status1:\n0: no any status(can''t run) -- 2020/03/10 lkchena\n1: ongoing  <-- wait drop -- 2020/02/28 lkchena\n2: change mold\n3: complete prepare job(change mold), wait run, lost   //sintering directly change to 3 w/o 2\n-- not use -- 4: test\n5: lost: complete test, ready to run <-- no lot track-in\n6: tool process, track-in with lot\n9: complete( 91: shift record, 99: shift complete )\n-1: delete by user  2018/10/22 lkchena\n\nstatus1=2 for change mold_id( before mfg run the order )\neng only can set 2, when no order(mfg/rd) is running --  2020/01/30 lkchena\n\n//for tool_order_bth -- 2020/02/15 lkchena\nstatus1:\n91: shift record -- 2019/01/02 lkchena\n99: shift complete order\n\nstatus is mysql keyword, so rename to status1 -- 2020/01/02 lkchena\n',
  `pri` int(11) DEFAULT 99 COMMENT 'link : pri @ erp_prod_stb_plan_bt\n2019/12/11 lkchena',
  `part_id` varchar(24) NOT NULL COMMENT 'part id',
  `part_raw` varchar(24) DEFAULT NULL COMMENT 'what is it ??? 2019/12/31 lkchea',
  `route_id` varchar(24) DEFAULT NULL,
  `part_name` varchar(24) DEFAULT NULL COMMENT '品名 -- 2019/12/31 lkchena',
  `part_erp_no` varchar(24) DEFAULT NULL COMMENT '料號 -- 2019/12/31 lkchena',
  `tool_id` varchar(12) NOT NULL DEFAULT '*' COMMENT 'forming tool 2020/04/28 lkchena',
  `tool_grp_id` varchar(24) DEFAULT NULL,
  `tool_grp` varchar(32) DEFAULT NULL,
  `mold_id` varchar(16) NOT NULL DEFAULT '*' COMMENT '??? 1 part vs 1 mold ???',
  `powder_type` varchar(16) DEFAULT NULL COMMENT 'powder type -- 2019/12/08 lkchena',
  `powder_erp_no` varchar(24) DEFAULT NULL COMMENT '粉料料號 -- 2019/12/31 lkchena',
  `lot_size_spec` int(11) NOT NULL DEFAULT 0 COMMENT 'total unit count 2018/06/06 lkchena\nin one cart / plate on the stb tool 2020/04/22 lkchena',
  `box_size_spec` int(11) NOT NULL DEFAULT 0 COMMENT 'Trinity maybe use it 2020/04/28 lkchena\ntotal cast count in one cart 2018/06/06 lkchena',
  `wafer_size_spec` int(11) NOT NULL DEFAULT 0 COMMENT 'Trinity maybe use it 2020/04/28 lkchena\n一個 cassette 可以裝多少個 part, 應該定義在 part table, 而不是 carrier 上\n<-- get data from wip table(from part table)',
  `cust_id` varchar(12) DEFAULT NULL,
  `cust_name` varchar(36) DEFAULT NULL,
  `user_id` varchar(16) DEFAULT NULL COMMENT 'default is NT account, if not, depend on user define(some employee don''t has NT account) -- 2017/11/04',
  `shift_id` varchar(4) DEFAULT NULL COMMENT 'maybe different shifts work with the mfg_no',
  `time_order` varchar(19) NOT NULL COMMENT 'order process day, so one order can have many mfg_no\n\nfor status=91, user can choose time_order -- 2020/01/02 lkchena',
  `time_start` varchar(19) DEFAULT NULL,
  `time_end` varchar(19) DEFAULT NULL,
  `time_fcst_end` varchar(19) DEFAULT NULL COMMENT 'use plc speed to forcast complete time - 2019/12/04 lkchena',
  `cnt_plan` int(11) NOT NULL DEFAULT 0,
  `cnt_cur` int(11) NOT NULL DEFAULT 0 COMMENT 'current count',
  `cnt_act` int(11) NOT NULL DEFAULT 0 COMMENT '實際驗收-pass qa -- 2020/02/14 lkchena',
  `cnt_ng` int(11) NOT NULL DEFAULT 0 COMMENT 'ng count - 2018/10/23 lkchena',
  `cnt_qc` int(11) NOT NULL DEFAULT 0,
  `cnt_test` int(11) NOT NULL DEFAULT 0 COMMENT '調機 -- 2020/02/14 lkchena',
  `cnt_tool` int(11) NOT NULL DEFAULT 0,
  `cnt_empty` int(11) NOT NULL DEFAULT 0 COMMENT '空打 ?',
  `shift_act` int(11) NOT NULL DEFAULT 0 COMMENT 'for status = 91, shift count -- 2020/01/02 lkchena',
  `shift_ng` int(11) NOT NULL DEFAULT 0,
  `shift_qc` int(11) NOT NULL DEFAULT 0,
  `shift_test` int(11) NOT NULL DEFAULT 0,
  `cum_start` int(11) NOT NULL DEFAULT 0 COMMENT 'tool cummulate number when mfg_order start 2020/02/13 lkchena',
  `cum_curr` int(11) NOT NULL DEFAULT 0 COMMENT 'current tool cummulate amount from plc - 2019/12/04 lkchena',
  `cum_act` int(11) NOT NULL DEFAULT 0 COMMENT 'cum_tool - cum_start',
  `cum_shift_act` int(11) NOT NULL DEFAULT 0 COMMENT 'tool_cum - last_shift_cum = qty_act_cum -- 2020/02/13 lkchena',
  `cum_shift_last` int(11) NOT NULL DEFAULT 0 COMMENT 'after get qty_tool_cum then whrite tool cum when take-over -- 2020/02/13 lkchena',
  `claim_memo` varchar(255) DEFAULT NULL,
  `comp_memo` varchar(64) DEFAULT NULL COMMENT 'complete / delete memo - 2018/10/23 lkchena',
  `comp_user_id` varchar(16) DEFAULT NULL,
  `comp_shift` varchar(4) DEFAULT NULL,
  `cad_file1` varchar(128) DEFAULT NULL COMMENT 'default: cad file',
  `cad_file2` varchar(128) DEFAULT NULL COMMENT 'reserved, cad file',
  `cad_file3` varchar(128) DEFAULT NULL COMMENT 'reserved,cad file',
  `date_stb` varchar(10) DEFAULT NULL,
  `date_due` varchar(10) DEFAULT NULL,
  `date_late` varchar(10) DEFAULT NULL COMMENT 'most late day to stb - 2019/12/04',
  `date_comp` varchar(10) DEFAULT NULL,
  `weight` float DEFAULT 0,
  `unit_label` int(11) DEFAULT 1 COMMENT 'weight unit: 1: kg 2: ton (tonnage) 2019/12/31 lkchena',
  `unit_type` int(11) DEFAULT 0 COMMENT '2020/05/02 lkchena\n0: by piece(default)\n1: by weigth\nothers.\n\nfn_get_wafer_id will create fullly wafer_id, by weight product will over limitation, so add this field',
  `owner_id` varchar(16) DEFAULT NULL COMMENT 'order owner, when something wrong, mfg can find sponsor -- 2019/12/31 lkchena',
  `owner_name` varchar(24) DEFAULT NULL,
  `claim_time` varchar(19) DEFAULT NULL,
  `note` varchar(128) DEFAULT NULL,
  `rec_user` varchar(16) DEFAULT NULL,
  `rec_time` varchar(19) DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`mfg_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `mfg_order_bt`
--

LOCK TABLES `mfg_order_bt` WRITE;
/*!40000 ALTER TABLE `mfg_order_bt` DISABLE KEYS */;
INSERT INTO `mfg_order_bt` VALUES ('MFG_NO-107','123441',0,0,1,'ACQ51F-1',NULL,'ROUTE-ZZZ-01',NULL,NULL,'AF15T01','FORMING-150T','150T成形','ACQ51F-1','H065M',NULL,2100,20,105,'000','',NULL,NULL,'2019/12/13 11:31:15',NULL,NULL,NULL,24000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'2019/11/25','2019/01/01',NULL,'',0,1,0,NULL,NULL,NULL,'','SYS','2019/12/13 11:31:15',NULL,'2020-04-28 08:49:53.456614');
/*!40000 ALTER TABLE `mfg_order_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:20
