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
-- Table structure for table `oee_tool_sch_bth`
--

DROP TABLE IF EXISTS `oee_tool_sch_bth`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `oee_tool_sch_bth` (
  `tool_id` varchar(12) NOT NULL COMMENT 'length extend to 12 for sintering "before/after" omi -- 2020/03/18 lkchena',
  `device` varchar(8) NOT NULL COMMENT '這個 table record "複合" status change\n\ndevice:\nMAIN -- 主機台\nLP_1~4 -- load_port 1~4\nCH_1~4 -- chamber 1~4',
  `status` varchar(16) DEFAULT NULL COMMENT '2017/07/21 lkchena:\nstatus is "複合", 因為 LP 會記錄 lot_id/cast_id/wafer_id,所以 extend length to 16\n\nstatus: up/down/pm/test/hold/mon/wcim/wmfg/off\nn/a --> just note, leave a message \n2017/06/26 lkchena',
  `pre_status` varchar(16) DEFAULT NULL COMMENT 'previous status',
  `status_sub` varchar(24) DEFAULT NULL COMMENT 'for user key-in sub category 2019/12/08 lkchena',
  `pre_status_sub` varchar(24) DEFAULT NULL COMMENT 'for user key-in sub category 2019/12/08 lkchena',
  `note` varchar(64) DEFAULT NULL COMMENT 'change note',
  `claim_time` varchar(19) NOT NULL,
  `claim_user` varchar(16) DEFAULT NULL,
  `mold_id` varchar(16) DEFAULT NULL,
  `recipe_id` varchar(32) DEFAULT NULL,
  `wph` float DEFAULT 0 COMMENT 'when status change, write in current recipe/wph, let eff easier to calculating 2019/02/06 lkchena',
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`tool_id`,`claim_time`,`device`),
  KEY `idx_tool_status_1_idx` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COMMENT='2017/06/26 lkchena:\nspc auto hold also put in this table';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `oee_tool_sch_bth`
--

LOCK TABLES `oee_tool_sch_bth` WRITE;
/*!40000 ALTER TABLE `oee_tool_sch_bth` DISABLE KEYS */;
/*!40000 ALTER TABLE `oee_tool_sch_bth` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:35
