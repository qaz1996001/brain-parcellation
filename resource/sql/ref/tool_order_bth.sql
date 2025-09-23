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
-- Table structure for table `tool_order_bth`
--

DROP TABLE IF EXISTS `tool_order_bth`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `tool_order_bth` (
  `tool_id` varchar(12) NOT NULL COMMENT 'length extend to 12 for sintering "before/after" omi -- 2020/03/18 lkchena',
  `tool_name` varchar(16) DEFAULT NULL COMMENT 'compatible current tool name - 2019/12/04 lkchena',
  `ws_type` varchar(12) DEFAULT '0' COMMENT 'change to use string, compatible with aruroal 2020/03/29 lkchena\n2020/03/17 lkchena\n',
  `tool_grp_id` varchar(24) DEFAULT NULL,
  `tool_grp` varchar(32) DEFAULT NULL,
  `mfg_no` varchar(16) NOT NULL COMMENT '??? why has mfg_no --> Trinity has the mfg_no, why ??? 2019/12/31 lkchena',
  `order_no` varchar(16) DEFAULT NULL COMMENT 'customer order - 2019/12/08 lkchena\nmaybe null(?)',
  `rd_flag` int(11) DEFAULT 0 COMMENT '0: prod 1: test run 2: rd \n2019/02/03 lkchena',
  `status1` int(11) DEFAULT 0 COMMENT 'status1: 0: wait run 1: ongoing 2:change mold 9: complete( 91: shift record, 99: shift complete ) -1: delete by user  2018/10/22 lkchena\n\nstatus1=2 for change mold_id( before mfg run the order )\neng only can set 2, when no order(mfg/rd) is running --  2020/01/30 lkchena\n\nstatus is mysql keyword, so rename to status1 -- 2020/01/02 lkchena\n\n//for tool_order_bth -- 2020/02/15 lkchena\nstatus1:\n91: shift record -- 2019/01/02 lkchena\n99: shift complete order',
  `pri` int(11) DEFAULT 99 COMMENT 'link : pri @ erp_prod_stb_plan_bt\n2019/12/11 lkchena',
  `mold_id` varchar(16) NOT NULL COMMENT '??? 1 part vs 1 mold ???',
  `recipe_id` varchar(32) DEFAULT NULL COMMENT 'mold param_id -- 2019/12/04 lkchena',
  `part_id` varchar(24) DEFAULT NULL COMMENT 'powder id',
  `part_raw` varchar(24) DEFAULT NULL COMMENT 'what is it ??? 2019/12/31 lkchea',
  `part_name` varchar(24) DEFAULT NULL COMMENT '品名 -- 2019/12/31 lkchena',
  `part_erp_no` varchar(24) DEFAULT NULL COMMENT '料號 -- 2019/12/31 lkchena',
  `powder_id` varchar(24) DEFAULT NULL COMMENT 'powder id -- 2019/12/08 lkchena',
  `powder_type` varchar(16) DEFAULT NULL COMMENT 'powder type -- 2019/12/08 lkchena',
  `powder_erp_no` varchar(24) DEFAULT NULL COMMENT '粉料料號 -- 2019/12/31 lkchena',
  `cust_id` varchar(12) DEFAULT NULL,
  `cust_name` varchar(36) DEFAULT NULL,
  `er` varchar(32) DEFAULT NULL,
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
  `lot_id` varchar(12) DEFAULT NULL COMMENT 'current lot_id - 2018/10/23 lkchena',
  `lot_qty` int(11) DEFAULT NULL COMMENT 'current lot_id qty - 2018/10/23 lkchena',
  `claim_memo` varchar(255) DEFAULT NULL,
  `comp_memo` varchar(64) DEFAULT NULL COMMENT 'complete / delete memo - 2018/10/23 lkchena',
  `comp_user_id` varchar(16) DEFAULT NULL,
  `comp_shift` varchar(4) DEFAULT NULL,
  `main_tool` varchar(12) DEFAULT NULL COMMENT 'measure tool -- 2019/03/13 lkchena',
  `meas_tool` varchar(12) DEFAULT NULL COMMENT 'measure tool -- 2019/03/13 lkchena',
  `cad_file1` varchar(128) DEFAULT NULL COMMENT 'default: cad file',
  `cad_file2` varchar(128) DEFAULT NULL COMMENT 'reserved, cad file',
  `cad_file3` varchar(128) DEFAULT NULL COMMENT 'reserved,cad file',
  `ope_no` varchar(7) NOT NULL COMMENT '2020/03/17 lkchena: for option flow, extend length to 7',
  `ope_name` varchar(36) DEFAULT NULL,
  `stage_id` varchar(16) NOT NULL,
  `stage_name` varchar(36) DEFAULT NULL,
  `stage_order` varchar(3) DEFAULT NULL,
  `date_stb` varchar(10) DEFAULT NULL,
  `date_due` varchar(10) DEFAULT NULL,
  `date_late` varchar(10) DEFAULT NULL COMMENT 'most late day to stb - 2019/12/04',
  `date_comp` varchar(10) DEFAULT NULL,
  `weight` float DEFAULT 0,
  `unit` int(11) DEFAULT 1 COMMENT 'weight unit: 1: kg 2: ton (tonnage) 2019/12/31 lkchena',
  `owner_id` varchar(16) DEFAULT NULL COMMENT 'order owner, when something wrong, mfg can find sponsor -- 2019/12/31 lkchena',
  `owner_name` varchar(24) DEFAULT NULL,
  `claim_time` varchar(19) DEFAULT NULL,
  `note` varchar(128) DEFAULT NULL,
  `rec_user` varchar(16) DEFAULT NULL,
  `rec_time` varchar(19) DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`tool_id`,`mfg_no`,`mold_id`,`time_order`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `tool_order_bth`
--

LOCK TABLES `tool_order_bth` WRITE;
/*!40000 ALTER TABLE `tool_order_bth` DISABLE KEYS */;
INSERT INTO `tool_order_bth` VALUES ('AF15T01',NULL,NULL,'FORMING-150T',NULL,'191213-001','123441',0,0,1,'ACQ51F-1',NULL,'ACQ51F-1',NULL,NULL,NULL,'','H065M',NULL,'000','',NULL,NULL,NULL,'2019/12/13 11:31:15',NULL,NULL,NULL,24000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'100.000','成型作業','Forming','成型站','','2019/11/25','2019/01/01',NULL,'',0,1,NULL,NULL,NULL,'','SYS','2019/12/13 11:31:15',NULL,'2020-03-17 02:23:02.582313'),('AF15T01',NULL,NULL,'FORMING-150T',NULL,'T-MFG_NO-003',NULL,1,1,33,'ACQ51F-1',NULL,'ACQ51F-1',NULL,NULL,NULL,'F123','C15M',NULL,NULL,NULL,NULL,'SYS','B','2020/02/15 20:30:20',NULL,NULL,NULL,778,0,12,0,0,0,0,0,12,0,0,0,0,0,-999999,0,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'100.000','成型作業','Forming','成型站','100','2019/12/24','2019/02/05',NULL,NULL,0,1,NULL,NULL,NULL,'*','SYS','2020/01/23 14:06:56',NULL,'2020-03-17 02:23:02.582313'),('AF15T01',NULL,NULL,'FORMING-150T',NULL,'T-MFG_NO-003',NULL,1,1,33,'ACQ51F-1',NULL,'ACQ51F-1',NULL,NULL,NULL,'F123','C15M',NULL,NULL,NULL,NULL,'SYS','B','2020/02/15 20:47:40',NULL,NULL,NULL,778,0,12,0,0,0,0,0,12,0,0,0,0,0,-999999,0,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'100.000','成型作業','Forming','成型站','100','2019/12/24','2019/02/05',NULL,NULL,0,1,NULL,NULL,NULL,'*','SYS','2020/01/23 14:06:56',NULL,'2020-03-17 02:23:02.582313'),('AF15T01',NULL,NULL,'FORMING-150T',NULL,'T-MFG_NO-003',NULL,1,99,33,'ACQ51F-1',NULL,'ACQ51F-1',NULL,NULL,NULL,'F123','C15M',NULL,NULL,NULL,NULL,'SYS','B','2020/02/15 20:50:48',NULL,NULL,NULL,778,0,12,0,0,0,0,0,12,0,0,0,0,0,-999999,0,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'100.000','成型作業','Forming','成型站','100','2019/12/24','2019/02/05',NULL,NULL,0,1,NULL,NULL,NULL,'*','SYS','2020/01/23 14:06:56',NULL,'2020-03-17 02:23:02.582313'),('AF15T01',NULL,NULL,'FORMING-150T',NULL,'T-MFG_NO-003',NULL,1,91,33,'ACQ51F-1',NULL,'ACQ51F-1',NULL,NULL,NULL,'F123','C15M',NULL,NULL,NULL,NULL,'SYS','B','2020/02/15 20:57:45',NULL,NULL,NULL,778,0,12,1,2,3,0,0,12,1,2,3,0,0,-999999,0,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'100.000','成型作業','Forming','成型站','100','2019/12/24','2019/02/05',NULL,NULL,0,1,NULL,NULL,NULL,'*','SYS','2020/01/23 14:06:56',NULL,'2020-03-17 02:23:02.582313'),('AF15T01',NULL,NULL,'FORMING-150T',NULL,'T-MFG_NO-003',NULL,1,91,33,'ACQ51F-1',NULL,'ACQ51F-1',NULL,NULL,NULL,'F123','C15M',NULL,NULL,NULL,NULL,'SYS','B','2020/02/15 20:58:58',NULL,NULL,NULL,778,0,112,11,22,33,0,0,100,10,20,30,0,0,-999999,0,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'100.000','成型作業','Forming','成型站','100','2019/12/24','2019/02/05',NULL,NULL,0,1,NULL,NULL,NULL,'*','SYS','2020/01/23 14:06:56',NULL,'2020-03-17 02:23:02.582313');
/*!40000 ALTER TABLE `tool_order_bth` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:17:07
