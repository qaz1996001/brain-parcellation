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
-- Table structure for table `dm_route_bt`
--

DROP TABLE IF EXISTS `dm_route_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `dm_route_bt` (
  `part_id` varchar(24) NOT NULL,
  `part_raw` varchar(16) DEFAULT NULL COMMENT '產品的原料: 旭宏 現場 習慣看 原料的編號 -- 2017/07/20 lkchena',
  `route_id` varchar(24) NOT NULL DEFAULT 'ROUTE-ZZZ-01' COMMENT 'for user not need to create each part / step record -- 2020/03/26 lkchena',
  `rd_flag` int(11) NOT NULL DEFAULT 0 COMMENT '2020/04/14 lkchena',
  `mold_id` varchar(16) DEFAULT NULL,
  `cust_id` varchar(12) DEFAULT NULL,
  `cust_name` varchar(36) DEFAULT NULL,
  `ope_no` varchar(7) NOT NULL COMMENT '2020/03/17 lkchena: for option flow, extend length to 7',
  `ope_name` varchar(36) DEFAULT NULL,
  `stage_id` varchar(16) DEFAULT NULL,
  `stage_name` varchar(36) DEFAULT NULL,
  `stage_order` varchar(3) DEFAULT NULL,
  `tool_grp_id` varchar(24) DEFAULT NULL,
  `tool_grp` varchar(32) DEFAULT NULL,
  `ws_type` varchar(12) NOT NULL DEFAULT '0' COMMENT 'change to use string, compatible with aruroal 2020/03/29 lkchena\n2020/03/17 lkchena\n',
  `tool_type1` varchar(12) NOT NULL DEFAULT '0' COMMENT 'for multi-step in one step, ex: 燒結乾振防鏽\n2020/03/22 lkchena',
  `tool_type2` varchar(12) NOT NULL DEFAULT '0' COMMENT 'for multi-step in one step, ex: 燒結乾振防鏽\n2020/03/22 lkchena',
  `tool_type3` varchar(12) NOT NULL DEFAULT '0' COMMENT 'for multi-step in one step, ex: 燒結乾振防鏽\n2020/03/22 lkchena',
  `in_out` int(11) NOT NULL DEFAULT 0 COMMENT 'default: 0: 廠內  1: 外注 1x: maybe A,B factory\n2020/03/25 lkchena',
  `area_id` varchar(12) DEFAULT NULL COMMENT '??? need this ???',
  `area_name` varchar(24) DEFAULT NULL,
  `proc_time` int(11) DEFAULT NULL COMMENT 'process time: unit: minutes',
  `cast_pcs_spec` int(11) DEFAULT NULL COMMENT '一個 cassette 可以裝多少個 part, 應該定義在 part table, 而不是 carrier 上\n<-- get data from wip table(from part table)',
  `rec_user` varchar(16) DEFAULT NULL COMMENT 'record user: NT account',
  `rec_time` datetime DEFAULT NULL COMMENT 'record time',
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`route_id`,`ope_no`),
  KEY `idx_route_ope_no` (`route_id`,`ope_no`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AVG_ROW_LENGTH=212 ROW_FORMAT=DYNAMIC COMMENT='1. the same table with dm_flow_bt, except the primary ke is different -- 2020/03/26 lkchena\r\n2. for user not need to create each part / step record -- 2020/03/26 lkchena';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `dm_route_bt`
--

LOCK TABLES `dm_route_bt` WRITE;
/*!40000 ALTER TABLE `dm_route_bt` DISABLE KEYS */;
INSERT INTO `dm_route_bt` VALUES ('ZZZ-02','ZZZ-02','ROUTE-ZZZ-01',0,'ACQ51F-1','Custom Real',NULL,'065.020','開始站點','STB','開始站','065','STB','STB','0','0','0','0',0,'START-1',NULL,1,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-02','ZZZ-02','ROUTE-ZZZ-01',0,'ACQ51F-1','Custom Real',NULL,'100.100','成型壓模','Forming','成型站','100','M-WET-1','M-WET-1','10','10','0','0',0,'C001-A',NULL,1,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-02','ZZZ-02','ROUTE-ZZZ-01',0,'ACQ51F-1','Custom Real',NULL,'200.050','Sorter - 換棧板','Sintering','燒結站','200','*','*','20','85','0','0',0,'S001',NULL,240,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-02','ZZZ-02','ROUTE-ZZZ-01',0,'ACQ51F-1','Custom Real',NULL,'200.100','燒結-主製程','Sintering','燒結站','200','SNT-8','SNT-8','20','20','0','0',0,'S001',NULL,240,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-02','ZZZ-02','ROUTE-ZZZ-01',0,'ACQ51F-1','Custom Real',NULL,'200.105','燒結-爐尾','Sintering','燒結站','200','SNT-8','SNT-8','21','21','0','0',0,'S001',NULL,240,32,'test','2017-07-23 00:00:00',NULL,'2020-04-21 06:03:36.027325'),('ZZZ-02','ZZZ-02','ROUTE-ZZZ-01',0,'ACQ51F-1','Custom Real',NULL,'200.220','燒結乾振防鏽','Sintering','燒結站','200','*','*','21','26','0','0',0,'S001',NULL,240,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-02','ZZZ-02','ROUTE-ZZZ-01',0,'ACQ51F-1','Custom Real',NULL,'200.250','燒結除油','Sintering','燒結站','200','*','*','21','60','0','0',0,'S001',NULL,240,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-02','ZZZ-02','ROUTE-ZZZ-01',0,'ACQ51F-1','Custom Real',NULL,'300.100','整形','Sizing','整形站','300','PRESSING-200T','PRESSING-200T','30','30','0','0',0,'P001-A',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-04-17 00:52:46.751010'),('ZZZ-02','ZZZ-02','ROUTE-ZZZ-01',0,'ACQ51F-1','Custom Real',NULL,'330.100','熱處理','Others','Others','999','Others','Others','0','0','0','0',0,'999',NULL,20,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-02','ZZZ-02','ROUTE-ZZZ-01',0,'','Custom Real',NULL,'500.100','鍛造','Forging','鍛造站','500','FORGING-TYPE-A','FORGING-TYPE-A','50','50','0','0',0,'G001',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-04-17 00:53:19.469757'),('ZZZ-02','ZZZ-02','ROUTE-ZZZ-01',0,'ACQ51F-1','Custom Real',NULL,'600.100','全檢','OQC','品檢站','600','Others','Others','0','0','0','0',0,'OQC-A',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-02','ZZZ-02','ROUTE-ZZZ-01',0,'ACQ51F-1','Custom Real',NULL,'600.300','音頻探傷機檢測','OQC','品檢站','600','Others','Others','0','0','0','0',0,'OQC-A',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-02','ZZZ-02','ROUTE-ZZZ-01',0,'ACQ51F-1','Custom Real',NULL,'860.100','雷刻','Marking','雷刻站','860','Others','Others','0','0','0','0',0,'Mark-1',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-04-28 02:09:32.323216'),('ZZZ-02','ZZZ-02','ROUTE-ZZZ-01',0,'ACQ51F-1','Custom Real',NULL,'888.100','包裝','Package','包裝站','888','Others','Others','0','0','0','0',0,'PACK-A',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-04-28 02:11:43.688555'),('ZZZ-02','ZZZ-02','ROUTE-ZZZ-01',0,'ACQ51F-1','Custom Real',NULL,'888.880','出貨完成','Package','包裝站','888','Others','Others','0','0','0','0',0,'PACK-A',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-04-28 02:11:43.689558'),('ZZZ-03','ZZZ-03','ROUTE-ZZZ-02',0,'ACQ51F-1','Custom Real',NULL,'065.020','開始站點','STB','開始站','065','STB','STB','0','0','0','0',0,'START-1',NULL,1,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-03','ZZZ-03','ROUTE-ZZZ-02',0,'ACQ51F-1','Custom Real',NULL,'100.100','成型壓模','Forming','成型站','100','M-WET-1','M-WET-1','10','10','0','0',0,'C001-A',NULL,1,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-03','ZZZ-03','ROUTE-ZZZ-02',0,'ACQ51F-1','Custom Real',NULL,'200.050','Sorter - 換棧板','Sintering','燒結站','200','*','*','20','85','0','0',0,'S001',NULL,240,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-03','ZZZ-03','ROUTE-ZZZ-02',0,'ACQ51F-1','Custom Real',NULL,'200.100','燒結-主製程','Sintering','燒結站','200','SNT-8','SNT-8','20','20','0','0',0,'S001',NULL,240,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-03','ZZZ-03','ROUTE-ZZZ-02',0,'ACQ51F-1','Custom Real',NULL,'200.105','燒結-爐尾','Sintering','燒結站','200','SNT-8','SNT-8','21','21','0','0',0,'S001',NULL,240,32,'test','2017-07-23 00:00:00',NULL,'2020-04-21 06:03:36.027325'),('ZZZ-03','ZZZ-03','ROUTE-ZZZ-02',0,'ACQ51F-1','Custom Real',NULL,'200.220','燒結乾振防鏽','Sintering','燒結站','200','*','*','21','26','0','0',0,'S001',NULL,240,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-03','ZZZ-03','ROUTE-ZZZ-02',0,'ACQ51F-1','Custom Real',NULL,'200.250','燒結除油','Sintering','燒結站','200','*','*','21','60','0','0',0,'S001',NULL,240,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-03','ZZZ-03','ROUTE-ZZZ-02',0,'ACQ51F-1','Custom Real',NULL,'300.100','整形','Sizing','整形站','300','PRESSING-200T','PRESSING-200T','30','30','0','0',0,'P001-A',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-04-17 00:52:46.751010'),('ZZZ-03','ZZZ-03','ROUTE-ZZZ-02',0,'ACQ51F-1','Custom Real',NULL,'330.100','熱處理','Others','Others','999','Others','Others','0','0','0','0',0,'999',NULL,20,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-03','ZZZ-03','ROUTE-ZZZ-02',0,'','Custom Real',NULL,'500.100','鍛造','Forging','鍛造站','500','FORGING-TYPE-A','FORGING-TYPE-A','50','50','0','0',0,'G001',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-04-17 00:53:19.469757'),('ZZZ-03','ZZZ-03','ROUTE-ZZZ-02',0,'ACQ51F-1','Custom Real',NULL,'600.100','全檢','OQC','品檢站','600','Others','Others','0','0','0','0',0,'OQC-A',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-03','ZZZ-03','ROUTE-ZZZ-02',0,'ACQ51F-1','Custom Real',NULL,'600.300','音頻探傷機檢測','OQC','品檢站','600','Others','Others','0','0','0','0',0,'OQC-A',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-03','ZZZ-03','ROUTE-ZZZ-02',0,'ACQ51F-1','Custom Real',NULL,'860.100','雷刻','Marking','雷刻站','860','Others','Others','0','0','0','0',0,'Mark-1',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-04-28 02:09:32.323216'),('ZZZ-03','ZZZ-03','ROUTE-ZZZ-02',0,'ACQ51F-1','Custom Real',NULL,'888.000','包裝','Package','包裝站','888','Others','Others','0','0','0','0',0,'PACK-A',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-04-28 02:14:04.802400'),('ZZZ-03','ZZZ-03','ROUTE-ZZZ-02',0,'ACQ51F-1','Custom Real',NULL,'888.880','出貨完成','Package','包裝站','888','Others','Others','0','0','0','0',0,'PACK-A',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-04-28 02:14:04.801880'),('ZZZ-04','ZZZ-04','ROUTE-ZZZ-03',0,'ACQ51F-1','Custom Real',NULL,'065.020','開始站點','STB','開始站','065','STB','STB','0','0','0','0',0,'START-1',NULL,1,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-04','ZZZ-04','ROUTE-ZZZ-03',0,'ACQ51F-1','Custom Real',NULL,'100.100','成型壓模','Forming','成型站','100','M-WET-1','M-WET-1','10','10','0','0',0,'C001-A',NULL,1,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-04','ZZZ-04','ROUTE-ZZZ-03',0,'ACQ51F-1','Custom Real',NULL,'200.050','Sorter - 換棧板','Sintering','燒結站','200','*','*','20','85','0','0',0,'S001',NULL,240,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-04','ZZZ-04','ROUTE-ZZZ-03',0,'ACQ51F-1','Custom Real',NULL,'200.100','燒結-主製程','Sintering','燒結站','200','SNT-8','SNT-8','20','20','0','0',0,'S001',NULL,240,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-04','ZZZ-04','ROUTE-ZZZ-03',0,'ACQ51F-1','Custom Real',NULL,'200.105','燒結-爐尾','Sintering','燒結站','200','SNT-8','SNT-8','21','21','0','0',0,'S001',NULL,240,32,'test','2017-07-23 00:00:00',NULL,'2020-04-21 06:03:36.028329'),('ZZZ-04','ZZZ-04','ROUTE-ZZZ-03',0,'ACQ51F-1','Custom Real',NULL,'200.220','燒結乾振防鏽','Sintering','燒結站','200','*','*','21','26','0','0',0,'S001',NULL,240,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-04','ZZZ-04','ROUTE-ZZZ-03',0,'ACQ51F-1','Custom Real',NULL,'200.250','燒結除油','Sintering','燒結站','200','*','*','21','60','0','0',0,'S001',NULL,240,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-04','ZZZ-04','ROUTE-ZZZ-03',0,'ACQ51F-1','Custom Real',NULL,'300.100','整形','Sizing','整形站','300','PRESSING-200T','PRESSING-200T','30','30','0','0',0,'P001-A',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-04-17 00:52:46.751010'),('ZZZ-04','ZZZ-04','ROUTE-ZZZ-03',0,'ACQ51F-1','Custom Real',NULL,'330.100','熱處理','Others','Others','999','Others','Others','0','0','0','0',0,'999',NULL,20,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-04','ZZZ-04','ROUTE-ZZZ-03',0,'','Custom Real',NULL,'500.100','鍛造','Forging','鍛造站','500','FORGING-TYPE-A','FORGING-TYPE-A','50','50','0','0',0,'G001',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-04-17 00:53:19.469757'),('ZZZ-04','ZZZ-04','ROUTE-ZZZ-03',0,'ACQ51F-1','Custom Real',NULL,'600.100','全檢','OQC','品檢站','600','Others','Others','0','0','0','0',0,'OQC-A',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-04','ZZZ-04','ROUTE-ZZZ-03',0,'ACQ51F-1','Custom Real',NULL,'600.300','音頻探傷機檢測','OQC','品檢站','600','Others','Others','0','0','0','0',0,'OQC-A',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-03-26 00:11:32.317590'),('ZZZ-04','ZZZ-04','ROUTE-ZZZ-03',0,'ACQ51F-1','Custom Real',NULL,'860.100','雷刻','Marking','雷刻站','860','Others','Others','0','0','0','0',0,'Mark-1',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-04-28 02:09:32.323216'),('ZZZ-04','ZZZ-04','ROUTE-ZZZ-03',0,'ACQ51F-1','Custom Real',NULL,'888.000','包裝','Package','包裝站','888','Others','Others','0','0','0','0',0,'PACK-A',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-04-28 02:14:44.465701'),('ZZZ-04','ZZZ-04','ROUTE-ZZZ-03',0,'ACQ51F-1','Custom Real',NULL,'888.880','出貨完成','Package','包裝站','888','Others','Others','0','0','0','0',0,'PACK-A',NULL,2,32,'test','2017-07-23 00:00:00',NULL,'2020-04-28 02:14:44.466203');
/*!40000 ALTER TABLE `dm_route_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:26
